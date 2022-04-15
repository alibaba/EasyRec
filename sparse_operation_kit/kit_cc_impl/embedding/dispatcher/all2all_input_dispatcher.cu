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

#include "common/include/forward_functions.h"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

struct IdenticalHash {
  using result_type = uint32_t;

  IdenticalHash() = default;

  template <typename TKey>
  static __device__ result_type compute(TKey const &key) {
    return static_cast<result_type>(key);
  }
};

/*It will dispatcher keys based on key % GPU_NUM */
template <typename KeyType, typename Hasher>
__global__ void selectKernel(KeyType const *input_keys, size_t num_keys, KeyType *output_keys,
                             uint32_t *output_indices, size_t chunks, size_t max_chunk_size,
                             uint32_t *chunk_sizes, const size_t ITEMS_PER_GPU_PER_WARP) {
  // set indices
  const size_t thread_cnt = blockDim.x * blockDim.y;
  const size_t stride_size = thread_cnt * gridDim.x;
  const size_t items_per_warp = chunks * ITEMS_PER_GPU_PER_WARP;
  const size_t items_per_block = KEY_WARPS_PER_BLOCK * items_per_warp;
  const size_t gpu_cnt_by_warps_cnt = chunks * KEY_WARPS_PER_BLOCK;
  int thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
  // set ptrs in smem
  extern __shared__ char smem[];
  KeyType *key_smem = (KeyType *)smem;
  uint32_t *idx_smem = (uint32_t *)(key_smem + items_per_block);
  uint32_t *cnt_smem = idx_smem + items_per_block;
  if (thread_idx < gpu_cnt_by_warps_cnt) {
    cnt_smem[thread_idx] = 0;
  }
  // if (thread_idx + blockIdx.x * thread_cnt < chunks) {
  //   chunk_sizes[thread_idx] = 0;
  // }
  __syncthreads();
  // do offset
  KeyType *curr_warp_key_smem = key_smem + threadIdx.y * items_per_warp;
  uint32_t *curr_warp_idx_smem = idx_smem + threadIdx.y * items_per_warp;
  uint32_t *curr_warp_cnt_smem = cnt_smem + threadIdx.y * chunks;
  uint32_t padded_input_size = (num_keys + warpSize - 1) / warpSize * warpSize;
  // loop on input_keys
  for (size_t idx = thread_idx + blockIdx.x * thread_cnt; idx < padded_input_size;
       idx += stride_size) {
    KeyType key = 0;
    size_t chunk_id = 0;
    uint32_t curr_local_idx = 0;
    uint32_t offset = 0;
    uint32_t is_full = 0;
    if (idx < num_keys) {
      key = input_keys[idx];
      chunk_id = Hasher::compute(key) % chunks;
      curr_local_idx = atomicAdd(curr_warp_cnt_smem + chunk_id, 1);
      offset = chunk_id * ITEMS_PER_GPU_PER_WARP + curr_local_idx;
      curr_warp_key_smem[offset] = key;
      curr_warp_idx_smem[offset] = idx;
    }
    is_full = (curr_local_idx == ITEMS_PER_GPU_PER_WARP - warpSize);
    uint32_t ballot_val = __ballot_sync(0xffffffff, is_full);
    // __syncwarp();
    int leading_zeros = __clz(ballot_val);
    while (leading_zeros < warpSize) {
      uint32_t full_gpu_idx = __shfl_sync(0xffffffff, chunk_id, warpSize - leading_zeros - 1);
      ballot_val &= (((uint32_t)0xffffffff) >> (leading_zeros + 1));
      leading_zeros = __clz(ballot_val);
      uint32_t curr_global_idx = 0;
      if (threadIdx.x == 0) {
        curr_global_idx = atomicAdd(chunk_sizes + full_gpu_idx, curr_warp_cnt_smem[full_gpu_idx]);
      }
      curr_global_idx = __shfl_sync(0xffffffff, curr_global_idx, 0);
      // __syncwarp();
      for (size_t output_idx = threadIdx.x; output_idx < curr_warp_cnt_smem[full_gpu_idx];
           output_idx += warpSize) {
        output_keys[full_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
            curr_warp_key_smem[full_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
        output_indices[full_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
            curr_warp_idx_smem[full_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
      }
      // __syncwarp();
    }
    __syncwarp();
    if (is_full) {
      curr_warp_cnt_smem[chunk_id] = 0;
    }
    __syncwarp();
  }
  // tail
  for (size_t has_gpu_idx = 0; has_gpu_idx < chunks; ++has_gpu_idx) {
    uint32_t curr_gpu_items = curr_warp_cnt_smem[has_gpu_idx];
    if (curr_gpu_items == 0) {
      continue;
    }
    uint32_t curr_global_idx = 0;
    if (threadIdx.x == 0) {
      curr_global_idx = atomicAdd(chunk_sizes + has_gpu_idx, curr_warp_cnt_smem[has_gpu_idx]);
    }
    curr_global_idx = __shfl_sync(0xffffffff, curr_global_idx, 0);
    for (size_t output_idx = threadIdx.x; output_idx < curr_warp_cnt_smem[has_gpu_idx];
         output_idx += warpSize) {
      output_keys[has_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
          curr_warp_key_smem[has_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
      output_indices[has_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
          curr_warp_idx_smem[has_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
    }
    __syncwarp();
  }
}

template <typename KeyType, typename ValueType>
class All2AllInputDispatcher : public Dispatcher {
 public:
  explicit All2AllInputDispatcher(ConstructionContext_t context)
      : Dispatcher(context),
        resource_mgr_(base_context()->get_resource_mgr()),
        num_keys_per_rank_(base_context()->get_replica_batch_size() *
                           base_context()->get_slot_num() * base_context()->get_nnz_per_slot()),
        ITEMS_PER_GPU_PER_WARP_(0) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    selected_keys_buf_.reserve(local_gpu_count);
    selected_indices_buf_.reserve(local_gpu_count);
    num_selected_keys_.reserve(local_gpu_count);
    num_exchanged_keys_.reserve(local_gpu_count);
    h_num_selected_keys_.reserve(local_gpu_count);
    h_num_exchanged_keys_.reserve(local_gpu_count);
    exchanged_keys_buf_.reserve(local_gpu_count);
    h_recv_chunk_offsets_.reserve(local_gpu_count);

    const size_t max_smem_size = resource_mgr_->get_local_gpu(0)->get_max_smem_size_per_sm();
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    ITEMS_PER_GPU_PER_WARP_ =
        max_smem_size - (sizeof(uint32_t) * KEY_WARPS_PER_BLOCK * global_gpu_count);
    ITEMS_PER_GPU_PER_WARP_ /=
        (global_gpu_count * KEY_WARPS_PER_BLOCK * (sizeof(KeyType) + sizeof(uint32_t)));
    ITEMS_PER_GPU_PER_WARP_ = (ITEMS_PER_GPU_PER_WARP_ / 16) * 16;
    if (ITEMS_PER_GPU_PER_WARP_ <= 33) {
      MESSAGE("[WARNING]: the performance in this device is not good enough..");
    }
  }

  void allocate_forward_spaces() override {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
      auto &buffer = base_context()->get_buffer(dev_id);
      auto &host_buffer = base_context()->get_host_buffer(dev_id);

      {
        Tensor2<KeyType> tensor;
        buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
        selected_keys_buf_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
        selected_indices_buf_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buffer->reserve({global_gpu_count}, &tensor);
        num_selected_keys_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buffer->reserve({global_gpu_count}, &tensor);
        num_exchanged_keys_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_num_selected_keys_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_num_exchanged_keys_.push_back(tensor);
      }
      {
        Tensor2<KeyType> tensor;
        buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
        exchanged_keys_buf_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        host_buffer->reserve({global_gpu_count + 1}, &tensor);
        h_recv_chunk_offsets_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }

  void allocate_backward_spaces() override {}

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    // step 1: reset count spaces.
    CK_CUDA(cudaMemsetAsync(num_selected_keys_[local_replica_id].get_ptr(), 0,
                            num_selected_keys_[local_replica_id].get_size_in_bytes(),
                            local_gpu->get_stream()));
    std::memset(h_recv_chunk_offsets_[local_replica_id].get_ptr(), 0,
                h_recv_chunk_offsets_[local_replica_id].get_size_in_bytes());

    // step 2: select keys for each GPU (rank)
    const auto &input_keys = replica_context->input("replica_values");
    {
      const size_t smem_size = local_gpu->get_max_smem_size_per_sm();
      CK_CUDA(cudaFuncSetAttribute(selectKernel<KeyType, IdenticalHash>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      size_t const grid_dim = local_gpu->get_sm_count();
      dim3 const block_dim(local_gpu->get_warp_size(), KEY_WARPS_PER_BLOCK);
      selectKernel<KeyType, IdenticalHash>
          <<<grid_dim, block_dim, smem_size, local_gpu->get_stream()>>>(
              /*input_keys=*/input_keys->GetPtrWithType<KeyType>(),
              /*num_keys=*/input_keys->get_num_elements(),
              /*output_keys=*/selected_keys_buf_[local_replica_id].get_ptr(),
              /*output_indices=*/selected_indices_buf_[local_replica_id].get_ptr(),
              /*chunks=*/global_gpu_count, /*max_chunk_size=*/num_keys_per_rank_,
              /*chunk_sizes=*/num_selected_keys_[local_replica_id].get_ptr(),
              /*ITEMS_PER_GPU_PER_WARP=*/ITEMS_PER_GPU_PER_WARP_);
      CK_CUDA(cudaGetLastError());
    }

    // step 3: exchange selected keys count among all GPUs
    CK_NCCL(ncclGroupStart());
    for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
      CK_NCCL(ncclSend(num_selected_keys_[local_replica_id].get_ptr() + dev_id, 1, ncclUint32,
                       /*peer=*/dev_id, local_gpu->get_nccl(), local_gpu->get_stream()));
      CK_NCCL(ncclRecv(num_exchanged_keys_[local_replica_id].get_ptr() + dev_id, 1, ncclUint32,
                       /*peer=*/dev_id, local_gpu->get_nccl(), local_gpu->get_stream()));
    }  // for dev_id in global_gpu_count
    CK_NCCL(ncclGroupEnd());

    // step 4: copy count from GPU to CPU and calculate count offsets
    CK_CUDA(cudaMemcpyAsync(h_num_selected_keys_[local_replica_id].get_ptr(),
                            num_selected_keys_[local_replica_id].get_ptr(),
                            num_selected_keys_[local_replica_id].get_size_in_bytes(),
                            cudaMemcpyDeviceToHost, local_gpu->get_stream()));
    CK_CUDA(cudaMemcpyAsync(h_num_exchanged_keys_[local_replica_id].get_ptr(),
                            num_exchanged_keys_[local_replica_id].get_ptr(),
                            num_exchanged_keys_[local_replica_id].get_size_in_bytes(),
                            cudaMemcpyDeviceToHost, local_gpu->get_stream()));
    CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));

    for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
      h_recv_chunk_offsets_[local_replica_id].get_ptr()[dev_id + 1] =
          h_recv_chunk_offsets_[local_replica_id].get_ptr()[dev_id] +
          h_num_exchanged_keys_[local_replica_id].get_ptr()[dev_id];
    }  // for dev_id in global_gpu_count

    // step 5: exchange selected keys among all GPUs
    CK_NCCL(ncclGroupStart());
    for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
      CK_NCCL(ncclSend(selected_keys_buf_[local_replica_id].get_ptr() + dev_id * num_keys_per_rank_,
                       h_num_selected_keys_[local_replica_id].get_ptr()[dev_id], GetNCCLType<KeyType>(),
                       /*peer=*/dev_id, local_gpu->get_nccl(), local_gpu->get_stream()));
      CK_NCCL(ncclRecv(exchanged_keys_buf_[local_replica_id].get_ptr() +
                           h_recv_chunk_offsets_[local_replica_id].get_ptr()[dev_id],
                       h_num_exchanged_keys_[local_replica_id].get_ptr()[dev_id], GetNCCLType<KeyType>(),
                       /*peer=*/dev_id, local_gpu->get_nccl(), local_gpu->get_stream()));
    }  // for dev_id in global_gpu_count
    CK_NCCL(ncclGroupEnd());

    // set output of this dispatcher
    replica_context->set_output("replica_exchanged_keys", exchanged_keys_buf_[local_replica_id]);
    replica_context->set_output("replica_h_recv_chunk_offsets",
                                h_recv_chunk_offsets_[local_replica_id]);
    replica_context->set_output("replica_h_num_exchanged_keys",
                                h_num_exchanged_keys_[local_replica_id]);
    replica_context->set_output("replica_h_num_selected_keys",
                                h_num_selected_keys_[local_replica_id]);
    replica_context->set_output("replica_num_selected_keys", num_selected_keys_[local_replica_id]);
    replica_context->set_output("replica_selected_indices_buf",
                                selected_indices_buf_[local_replica_id]);
  }

  void backward(const Context_t &replica_context) override {}

 private:
  const std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t num_keys_per_rank_;
  size_t ITEMS_PER_GPU_PER_WARP_;

  // forward spaces
  Tensors2<KeyType> selected_keys_buf_;
  Tensors2<uint32_t> selected_indices_buf_;
  Tensors2<uint32_t> num_selected_keys_;
  Tensors2<uint32_t> num_exchanged_keys_;
  Tensors2<uint32_t> h_num_selected_keys_;
  Tensors2<uint32_t> h_num_exchanged_keys_;
  Tensors2<KeyType> exchanged_keys_buf_;
  Tensors2<uint32_t> h_recv_chunk_offsets_;
};

REGISTER_INPUT_DISPATCHER_BUILDER("All2AllInput", 
                                  DataType::Int64,
                                  DataType::Float32, 
                                  All2AllInputDispatcher<int64_t, float>);
REGISTER_INPUT_DISPATCHER_BUILDER("All2AllInput", 
                                  DataType::Int64,
                                  DataType::Float16, 
                                  All2AllInputDispatcher<int64_t, __half>);
REGISTER_INPUT_DISPATCHER_BUILDER("All2AllInput", 
                                  DataType::Uint32,
                                  DataType::Float32, 
                                  All2AllInputDispatcher<uint32_t, float>);
REGISTER_INPUT_DISPATCHER_BUILDER("All2AllInput", 
                                  DataType::Uint32,
                                  DataType::Float16, 
                                  All2AllInputDispatcher<uint32_t, __half>);

}  // namespace SparseOperationKit
