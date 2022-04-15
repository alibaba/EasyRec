/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/include/dumping_functions.h"

#include <cstdint>

#include "common.h"
#include "common/include/forward_functions.h"

namespace SparseOperationKit {

// distribute keys to GPU based on key % GPU_NUM
template <typename KeyType>
void save_params_helper(const std::shared_ptr<ParamInterface> &param,
                        const std::shared_ptr<ResourcesManager> &resource_mgr,
                        std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
                        size_t &num_total_keys) {
  // this lookup distribute keys to GPU based on key % GPU_NUM
  const size_t local_gpu_count = resource_mgr->get_local_gpu_count();
  const size_t worker_num = resource_mgr->get_workers_num();
  const size_t worker_id = resource_mgr->get_worker_id();
  const size_t embedding_vector_size = param->get_embedding_vec_size();

  // step 1: get the count of key-index pairs on each hashtable on local worker
  std::unique_ptr<size_t[]> count;
  size_t local_worker_max_count = 0ul;
  if (0 == worker_id) {  // chief worker
    count.reset(new size_t[worker_num * local_gpu_count]());
  } else {  // non-chief worker
    count.reset(new size_t[local_gpu_count]());
  }
  HugeCTR::CudaDeviceContext device_context;
  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    device_context.set_device(local_gpu->get_local_device_id());

    const auto &hashtable = param->get_hashtable(dev_id);
    const size_t hashtable_size = hashtable->get_size(local_gpu->get_stream());
    if (hashtable_size != hashtable->get_value_head(local_gpu->get_stream()))
      throw std::runtime_error(ErrorBase + " hashtable get_value_head() not equal to get_size().");
    if (hashtable_size > param->get_max_vocabulary_size_per_gpu())
      throw std::runtime_error(ErrorBase + " keys count on GPU: " + std::to_string(dev_id) +
                               " is out of the range of max_vocabulary_size_per_gpu.");

    count[dev_id] = hashtable_size;
    MESSAGE("Worker: " + std::to_string(worker_id) + ", GPU: " + std::to_string(dev_id) +
            " key-index count = " + std::to_string(hashtable_size));
    local_worker_max_count =
        (local_worker_max_count > count[dev_id]) ? local_worker_max_count : count[dev_id];
  }  // for dev_id in local_gpu_count

  // step 2: gather count among all workers
  size_t *d_global_max_count = nullptr;
  std::vector<size_t *> d_count(local_gpu_count);
  size_t *d_count_aggregation = nullptr;
  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    device_context.set_device(local_gpu->get_local_device_id());

    if (0 == worker_id && 0 == dev_id) {  // chief worker and chief GPU
      CK_CUDA(cudaMalloc(&d_global_max_count, sizeof(size_t) * 1));
      CK_CUDA(cudaMalloc(&d_count_aggregation, sizeof(size_t) * worker_num * local_gpu_count));
    }
    CK_CUDA(cudaMalloc(&d_count[dev_id], sizeof(size_t) * 1));
    CK_CUDA(cudaMemcpyAsync(d_count[dev_id], &count[dev_id], sizeof(size_t) * 1,
                            cudaMemcpyHostToDevice, local_gpu->get_stream()));
  }  // for dev_id in local_gpu_count

  resource_mgr->sync_all_workers();

  CK_NCCL(ncclGroupStart());
  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    CK_NCCL(ncclReduce(d_count[dev_id], d_global_max_count, 1, ncclUint64, ncclMax,
                       /*root=*/0, local_gpu->get_nccl(), local_gpu->get_stream()));
  }  // for dev_id in local_gpu_count
  CK_NCCL(ncclGroupEnd());

  CK_NCCL(ncclGroupStart());
  if (0 == worker_id) {  // chief worker
    const auto &local_gpu = resource_mgr->get_local_gpu(0);
    device_context.set_device(local_gpu->get_local_device_id());
    for (size_t rank = 0; rank < resource_mgr->get_global_gpu_count(); rank++) {
      CK_NCCL(ncclRecv(d_count_aggregation + rank, 1, ncclUint64, /*peer=*/rank,
                       local_gpu->get_nccl(), local_gpu->get_stream()));
    }  // for rank in global_gpu_count
  }
  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    CK_NCCL(ncclSend(d_count[dev_id], 1, ncclUint64, /*peer=*/0, local_gpu->get_nccl(),
                     local_gpu->get_stream()));
  }  // for dev_id in local_gpu_count
  CK_NCCL(ncclGroupEnd());

  if (0 == worker_id) {  // chief worker
    const auto &local_gpu = resource_mgr->get_local_gpu(0);
    device_context.set_device(local_gpu->get_local_device_id());

    // in cheif worker, the local_worker_max_count equals to global_worker_max_count
    local_worker_max_count = 0;
    CK_CUDA(cudaMemcpyAsync(&local_worker_max_count, d_global_max_count, sizeof(size_t) * 1,
                            cudaMemcpyDeviceToHost, local_gpu->get_stream()));

    // in cheif worker, count array contains the number of valid keys for each GPU.
    CK_CUDA(cudaMemcpyAsync(count.get(), d_count_aggregation,
                            sizeof(size_t) * worker_num * local_gpu_count, cudaMemcpyDeviceToHost,
                            local_gpu->get_stream()));
    CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));
  }  // chief worker

  // step 3: allocate temp spaces for dump parameters from GPU to CPU
  std::unique_ptr<KeyType *[]> d_hash_table_key(new KeyType *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
  std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
  for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    device_context.set_device(local_gpu->get_local_device_id());

    CK_CUDA(cudaMalloc(&d_hash_table_key[dev_id], local_worker_max_count * sizeof(KeyType)));
    CK_CUDA(cudaMalloc(&d_hash_table_value_index[dev_id], local_worker_max_count * sizeof(size_t)));
    CK_CUDA(cudaMalloc(&d_hash_table_value[dev_id],
                       local_worker_max_count * embedding_vector_size * sizeof(float)));
    CK_CUDA(cudaMalloc(&d_dump_counter[dev_id], 1 * sizeof(size_t)));
  }  // for dev_id in local_gpu_count
  resource_mgr->sync_all_workers();

  // step 4: dump parameters to temp spaces
  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    if (0 == count[dev_id]) continue;
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    device_context.set_device(local_gpu->get_local_device_id());
    MESSAGE("Worker: " + std::to_string(worker_id) + ", GPU: " + std::to_string(dev_id) +
            ": dumping parameters from hashtable..");

    // get hashtable key-index pairs.
    const auto &hashtable = param->get_hashtable(dev_id);
    if (!hashtable->identical_mapping()) {  // hashtable contains internal states
      hashtable->dump(d_hash_table_key[dev_id], d_hash_table_value_index[dev_id],
                      d_dump_counter[dev_id], local_gpu->get_stream());

      // get embedding vector by sorted index
      get_hash_value(count[dev_id], embedding_vector_size, d_hash_table_value_index[dev_id],
                     param->get_embedding_table_tensor(dev_id)->GetPtrWithType<float>(),
                     d_hash_table_value[dev_id], local_gpu->get_stream());
    } else {  // hashtable does not contain internal states.
      const size_t hashtable_size = hashtable->get_size(local_gpu->get_stream());
      // generate keys for this GPU
      generate_dummy_keys(d_hash_table_key[dev_id],
                          /*num_keys=*/hashtable_size,
                          /*global_replica_id=*/local_gpu->get_global_device_id(),
                          /*global_gpu_count=*/resource_mgr->get_global_gpu_count(),
                          local_gpu->get_stream());
      CK_CUDA(cudaGetLastError());

      // copy embedding values from embedding_table_tensor
      CK_CUDA(cudaMemcpyAsync(d_hash_table_value[dev_id],
                              param->get_embedding_table_tensor(dev_id)->GetPtrWithType<float>(),
                              sizeof(float) * hashtable_size * embedding_vector_size,
                              cudaMemcpyDeviceToDevice, local_gpu->get_stream()));
    }  // get keys & embedding_values
  }    // for dev_id in local_gpu_count

  // step 6: aggregate param from non-cheif to chief worker, only cheif worker
  // needs to do GPU->CPU memory copy, other workers only send their data to
  // cheif worker via NCCL.
  if (0 == worker_id) {  // cheif worker
    size_t host_num_keys_offset = 0ul;

    for (size_t worker = 0; worker < worker_num; worker++) {
      if (worker_id != worker) { /*cheif worker receives data from other workers*/
        CK_NCCL(ncclGroupStart());
        for (size_t recv_worker = 1; recv_worker < worker_num; recv_worker++) {
          if (worker == recv_worker) { /*cheif worker receives valid data from other worker*/
            for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
              const int32_t peer = worker * local_gpu_count + dev_id;
              const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
              const size_t pair_count = count[recv_worker * local_gpu_count + dev_id];
              CK_NCCL(ncclRecv(d_hash_table_key[dev_id], pair_count, GetNCCLType<KeyType>(), /*peer=*/peer,
                               local_gpu->get_nccl(), local_gpu->get_stream()));
              CK_NCCL(ncclRecv(d_hash_table_value[dev_id], pair_count * embedding_vector_size,
                               ncclFloat32, /*peer=*/peer, local_gpu->get_nccl(),
                               local_gpu->get_stream()));
            }  // for dev_id in local_gpu_count
            MESSAGE("Worker: " + std::to_string(worker) + "'s data is received by cheif node.");
          } else { /*cheif worker receives dummy data from other worker*/
            for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
              CK_NCCL(ncclRecv(d_count[dev_id], 1, ncclUint64,
                               /*peer=*/recv_worker * local_gpu_count + dev_id,
                               resource_mgr->get_local_gpu(dev_id)->get_nccl(),
                               resource_mgr->get_local_gpu(dev_id)->get_stream()));
            }  // for dev_id in local_gpu_count
          }
        }  // for recv_worker in [1, worker_num)
        CK_NCCL(ncclGroupEnd());
      } /*cheif worker receives data from other workers*/

      /*cheif worker copy data from GPU to CPU output tensor*/
      for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        const size_t pair_count = count[worker * local_gpu_count + dev_id];
        const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
        CK_CUDA(cudaMemcpyAsync(keys->GetPtrWithType<KeyType>() + host_num_keys_offset,
                                d_hash_table_key[dev_id], pair_count * sizeof(KeyType),
                                cudaMemcpyDeviceToHost, local_gpu->get_stream()));
        CK_CUDA(cudaMemcpyAsync(embedding_values->GetPtrWithType<float>() +
                                    host_num_keys_offset * embedding_vector_size,
                                d_hash_table_value[dev_id],
                                pair_count * embedding_vector_size * sizeof(float),
                                cudaMemcpyDeviceToHost, local_gpu->get_stream()));
        host_num_keys_offset += pair_count;
      }  // for dev_id in local_gpu_count
      resource_mgr->sync_local_gpus();
    }  // for worker in worker_num
    num_total_keys = host_num_keys_offset;

  } else {  // non-cheif worker
    for (size_t worker = 1; worker < worker_num; worker++) {
      if (worker == worker_id) { /*sub worker send valid data to cheif worker*/
        CK_NCCL(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
          const size_t pair_count = count[dev_id];
          const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
          const int32_t peer = 0 * local_gpu_count + dev_id;
          CK_NCCL(ncclSend(d_hash_table_key[dev_id], pair_count, GetNCCLType<KeyType>(),
                           /*peer=*/peer, local_gpu->get_nccl(), local_gpu->get_stream()));
          CK_NCCL(ncclSend(d_hash_table_value[dev_id], pair_count * embedding_vector_size,
                           ncclFloat32, /*peer=*/peer, local_gpu->get_nccl(),
                           local_gpu->get_stream()));
        }  // for dev_id in local_gpu_count
        CK_NCCL(ncclGroupEnd());
        MESSAGE("Worker: " + std::to_string(worker) + "'s data sent to cheif worker.");
      } else { /*sub worker send dummy data to cheif worker*/
        CK_NCCL(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
          CK_NCCL(ncclSend(d_count[dev_id], 1, ncclUint64,
                           /*peer=*/0 * local_gpu_count + dev_id,
                           resource_mgr->get_local_gpu(dev_id)->get_nccl(),
                           resource_mgr->get_local_gpu(dev_id)->get_stream()));
        }
        CK_NCCL(ncclGroupEnd());
      }
    }  // for worker in [1, worker_num)
  }    // non-chief worker

  // step 7: synchonize all workers
  resource_mgr->sync_all_workers();

  // finally: release temp spaces
  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    const auto &local_gpu = resource_mgr->get_local_gpu(dev_id);
    device_context.set_device(local_gpu->get_local_device_id());

    if (0 == worker_id && 0 == dev_id) {
      CK_CUDA(cudaFree(d_global_max_count));
      CK_CUDA(cudaFree(d_count_aggregation));
    }
    CK_CUDA(cudaFree(d_count[dev_id]));
    CK_CUDA(cudaFree(d_hash_table_key[dev_id]));
    CK_CUDA(cudaFree(d_hash_table_value_index[dev_id]));
    CK_CUDA(cudaFree(d_hash_table_value[dev_id]));
    CK_CUDA(cudaFree(d_dump_counter[dev_id]));
  }  // for dev_id in local_gpu_count
}

template void save_params_helper<int64_t>(
      const std::shared_ptr<ParamInterface> &param,
      const std::shared_ptr<ResourcesManager> &resource_mgr,
      std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
      size_t &num_total_keys);
template void save_params_helper<uint32_t>(
      const std::shared_ptr<ParamInterface> &param,
      const std::shared_ptr<ResourcesManager> &resource_mgr,
      std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
      size_t &num_total_keys);

template <typename KeyType>
void restore_params_helper(std::shared_ptr<ParamInterface> &param,
                           const std::shared_ptr<ResourcesManager> &resource_mgr,
                           const std::shared_ptr<Tensor> &keys,
                           const std::shared_ptr<Tensor> &embedding_values,
                           const size_t num_total_keys) {
  const size_t total_max_vocabulary_size =
      param->get_max_vocabulary_size_per_gpu() * resource_mgr->get_global_gpu_count();

  MESSAGE("num_total_keys = " + std::to_string(num_total_keys) +
          ", "
          "while total_max_vocabulary_size = " +
          std::to_string(total_max_vocabulary_size));

  const KeyType *key_ptr = keys->GetPtrWithType<KeyType>();
  const float *embedding_ptr = embedding_values->GetPtrWithType<float>();

  // step 1: allocate temporary spaces
  const size_t worker_id = resource_mgr->get_worker_id();
  const size_t local_gpu_count = resource_mgr->get_local_gpu_count();
  constexpr size_t chunk_size = 1000;
  constexpr size_t hash_table_key_tile_size = 1;
  const size_t embedding_vec_size = param->get_embedding_vec_size();
  const size_t hash_table_key_tile_size_in_bytes = hash_table_key_tile_size * sizeof(KeyType);
  const size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_size;
  const size_t hash_table_key_chunk_size_in_bytes = hash_table_key_chunk_size * sizeof(KeyType);
  const size_t hash_table_value_index_chunk_size_in_bytes =
      hash_table_key_chunk_size * sizeof(size_t);
  const size_t hash_table_value_tile_size = embedding_vec_size;
  const size_t hash_table_value_tile_size_in_bytes = hash_table_value_tile_size * sizeof(float);
  const size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_size;
  const size_t hash_table_value_chunk_size_in_bytes = hash_table_value_chunk_size * sizeof(float);

  // cannot decide precise the number of values for each GPU, so allocate enough spaces
  std::unique_ptr<size_t[]> tile_counter_per_gpu(new size_t[local_gpu_count]());
  std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]());
  std::unique_ptr<size_t *[]> d_hash_table_value_index_chunk_per_gpu(new size_t *[local_gpu_count]);
  std::unique_ptr<KeyType *[]> h_hash_table_key_chunk_per_gpu(new KeyType *[local_gpu_count]);
  std::unique_ptr<KeyType *[]> d_hash_table_key_chunk_per_gpu(new KeyType *[local_gpu_count]);
  std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);

  HugeCTR::CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_mgr->get_local_gpu(id);
    context.set_device(local_gpu->get_local_device_id());

    CK_CUDA(cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id],
                       hash_table_value_index_chunk_size_in_bytes));
    CK_CUDA(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                            hash_table_value_index_chunk_size_in_bytes, local_gpu->get_stream()));
    CK_CUDA(
        cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_bytes));
    CK_CUDA(cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_bytes));
    CK_CUDA(cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id],
                           hash_table_value_chunk_size_in_bytes));
  }  // for id in local_gpu_count
  resource_mgr->sync_local_gpus();

  // step 2: do uploading
  size_t loop_num = num_total_keys / chunk_size;
  MESSAGE("Worker " + std::to_string(worker_id) +
          ": Start uploading parameters. "
          "Total loop_num = " +
          std::to_string(loop_num));
  for (size_t i = 0; i < loop_num; i++) {
    KeyType *key_dst_buf;
    float *value_dst_buf;
    for (size_t k = 0; k < chunk_size; k++) {
      const KeyType key = key_ptr[i * chunk_size + k];
      const size_t global_gpu_id = key % resource_mgr->get_global_gpu_count();
      const size_t local_gpu_id = resource_mgr->cal_local_id_from_global_id(global_gpu_id);
      const size_t dst_worker = resource_mgr->cal_worker_id_from_global_id(global_gpu_id);
      if (dst_worker == worker_id) {  // it belongs to this worker
        key_dst_buf = h_hash_table_key_chunk_per_gpu[local_gpu_id] +
                      tile_counter_in_chunk_per_gpu[local_gpu_id] * hash_table_key_tile_size;
        *key_dst_buf = key;

        value_dst_buf = h_hash_table_value_chunk_per_gpu[local_gpu_id] +
                        tile_counter_in_chunk_per_gpu[local_gpu_id] * hash_table_value_tile_size;
        std::memcpy(value_dst_buf, embedding_ptr + (i * chunk_size + k) * embedding_vec_size,
                    hash_table_value_tile_size_in_bytes);
        tile_counter_in_chunk_per_gpu[local_gpu_id]++;
      } else {
        continue;
      }
    }  // for k in chunk_size

    // insert to hash table
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_mgr->get_local_gpu(id);
      context.set_device(local_gpu->get_local_device_id());

      const size_t tile_count = tile_counter_in_chunk_per_gpu[id];
      CK_CUDA(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                              h_hash_table_key_chunk_per_gpu[id], tile_count * sizeof(KeyType),
                              cudaMemcpyHostToDevice, local_gpu->get_stream()));

      const size_t value_index_offset = tile_counter_per_gpu[id];
      size_t *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
      if (tile_count > 0) {
        memset_liner(value_index_buf, value_index_offset, 1ul, tile_count, local_gpu->get_stream());
      }

      // do hash table insert <key, index>
      param->get_hashtable(id)->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                       tile_count, local_gpu->get_stream());
      param->get_hashtable(id)->get_and_add_value_head(tile_count, local_gpu->get_stream());
    }  // for id in local_gpu_count

    // copy embedding vectors
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_mgr->get_local_gpu(id);
      context.set_device(local_gpu->get_local_device_id());

      const size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * embedding_vec_size;
      const size_t value_chunk_offset = tile_counter_per_gpu[id] * embedding_vec_size;
      const float *src_buf = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf =
          param->get_embedding_table_tensor(id)->GetPtrWithType<float>() + value_chunk_offset;
      CK_CUDA(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                              cudaMemcpyHostToDevice, local_gpu->get_stream()));
    }  // for id in local_gpu_count

    resource_mgr->sync_local_gpus();

    // set counter value
    for (size_t id = 0; id < local_gpu_count; id++) {
      tile_counter_per_gpu[id] += tile_counter_in_chunk_per_gpu[id];
      tile_counter_in_chunk_per_gpu[id] = 0;
      if (tile_counter_per_gpu[id] > param->get_max_vocabulary_size_per_gpu())
        throw std::runtime_error(ErrorBase + "The size of hash table on GPU " + std::to_string(id) +
                                 " is out of range " +
                                 std::to_string(param->get_max_vocabulary_size_per_gpu()));
    }  // for id in local_gpu_count
  }    // for i in loop_num

  // step 3: process the remaining data (less than a chunk)
  const size_t remain_loop_num = num_total_keys - loop_num * chunk_size;
  KeyType *key_dst_buf;
  size_t *value_index_buf;
  float *value_dst_buf;
  for (size_t i = 0; i < remain_loop_num; i++) {
    const KeyType key = key_ptr[loop_num * chunk_size + i];
    const size_t global_gpu_id = key % resource_mgr->get_global_gpu_count();
    const size_t local_gpu_id = resource_mgr->cal_local_id_from_global_id(global_gpu_id);
    const size_t dst_worker = resource_mgr->cal_worker_id_from_global_id(global_gpu_id);

    if (worker_id == dst_worker) {
      const auto &local_gpu = resource_mgr->get_local_gpu(local_gpu_id);
      context.set_device(local_gpu->get_local_device_id());

      // copy hashtable key from CPU to GPU
      key_dst_buf = d_hash_table_key_chunk_per_gpu[local_gpu_id];
      CK_CUDA(cudaMemcpyAsync(key_dst_buf, &key, hash_table_key_tile_size_in_bytes,
                              cudaMemcpyHostToDevice, local_gpu->get_stream()));

      // set value_index
      const size_t value_index_offset = tile_counter_per_gpu[local_gpu_id];
      value_index_buf = d_hash_table_value_index_chunk_per_gpu[local_gpu_id];
      memset_liner(value_index_buf, value_index_offset, 1ul, 1ul, local_gpu->get_stream());

      // hashtable insert
      param->get_hashtable(local_gpu_id)
          ->insert(d_hash_table_key_chunk_per_gpu[local_gpu_id], value_index_buf,
                   hash_table_key_tile_size, local_gpu->get_stream());
      param->get_hashtable(local_gpu_id)
          ->get_and_add_value_head(hash_table_key_tile_size, local_gpu->get_stream());

      // memcpy embeddding vectors
      const size_t value_offset = tile_counter_per_gpu[local_gpu_id] * embedding_vec_size;
      value_dst_buf =
          param->get_embedding_table_tensor(local_gpu_id)->GetPtrWithType<float>() + value_offset;
      CK_CUDA(cudaMemcpyAsync(
          value_dst_buf, embedding_ptr + (loop_num * chunk_size + i) * embedding_vec_size,
          hash_table_value_tile_size_in_bytes, cudaMemcpyHostToDevice, local_gpu->get_stream()));

      // set counter
      tile_counter_per_gpu[local_gpu_id] += hash_table_key_tile_size;
    } else {
      continue;
    }

    resource_mgr->sync_local_gpus();
  }  // for i in remain_loop_num

  resource_mgr->sync_all_workers();

  // finally: release temp spaces
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(resource_mgr->get_local_gpu(id)->get_local_device_id());

    CK_CUDA(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
    CK_CUDA(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
    CK_CUDA(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
    CK_CUDA(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
  }
}

template void restore_params_helper<int64_t>(
    std::shared_ptr<ParamInterface> &param,
    const std::shared_ptr<ResourcesManager> &resource_mgr,
    const std::shared_ptr<Tensor> &keys,
    const std::shared_ptr<Tensor> &embedding_values,
    const size_t num_total_keys);
template void restore_params_helper<uint32_t>(
    std::shared_ptr<ParamInterface> &param,
    const std::shared_ptr<ResourcesManager> &resource_mgr,
    const std::shared_ptr<Tensor> &keys,
    const std::shared_ptr<Tensor> &embedding_values,
    const size_t num_total_keys);
      

}  // namespace SparseOperationKit