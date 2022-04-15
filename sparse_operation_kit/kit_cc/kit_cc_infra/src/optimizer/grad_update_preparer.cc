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

#include "optimizer/grad_update_preparer.h"

#include <cmath>

#include "optimizer/prepare_functions.h"
#include "optimizer/update_functions.h"
#include "tensor_buffer/tensor2_wrapper.h"

namespace SparseOperationKit {

GradUpdatePreparer::GradUpdatePreparer(
    const size_t global_batch_size, const size_t max_feature_num,
    const size_t max_vocabulary_size_per_gpu,
    std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>& host_buffers,
    std::shared_ptr<ResourcesManager>& resource_mgr)
    : global_batch_size_(global_batch_size),
      max_feature_num_(max_feature_num),
      max_vocabulary_size_per_gpu_(max_vocabulary_size_per_gpu),
      buffers_(buffers),
      host_buffers_(host_buffers),
      resource_mgr_(resource_mgr) {
  const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();

  positions_.reserve(local_gpu_count);
  sort_positions_.reserve(local_gpu_count);
  sort_positions_interface_.reserve(local_gpu_count);
  sort_duplicated_indices_.reserve(local_gpu_count);
  sort_duplicated_indices_interface_.reserve(local_gpu_count);
  indices_unique_flags_.reserve(local_gpu_count);
  indices_unique_sums_.reserve(local_gpu_count);
  indices_unique_indexes_.reserve(local_gpu_count);
  indices_unique_indexes_interface_.reserve(local_gpu_count);
  dev_unique_indices_nums_.reserve(local_gpu_count);
  host_unique_indices_nums_.reserve(local_gpu_count);
  temp_storage_sort_tensors_.reserve(local_gpu_count);
  temp_storage_sum_tensors_.reserve(local_gpu_count);

  for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
    {
      Tensor2<int64_t> tensor;
      buffers_[dev_id]->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
      positions_.push_back(tensor);
    }
    {
      Tensor2<int64_t> tensor;
      buffers_[dev_id]->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
      sort_positions_.push_back(tensor);
      sort_positions_interface_.push_back(Tensor2Wrapper<int64_t>::create(tensor));
    }
    {
      Tensor2<int64_t> tensor;
      buffers_[dev_id]->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
      sort_duplicated_indices_.push_back(tensor);
      sort_duplicated_indices_interface_.push_back(Tensor2Wrapper<int64_t>::create(tensor));
    }
    {
      Tensor2<uint32_t> tensor;
      buffers_[dev_id]->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
      indices_unique_flags_.push_back(tensor);
    }
    {
      Tensor2<uint32_t> tensor;
      buffers_[dev_id]->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
      indices_unique_sums_.push_back(tensor);
    }
    {
      Tensor2<size_t> tensor;
      buffers_[dev_id]->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
      indices_unique_indexes_.push_back(tensor);
      indices_unique_indexes_interface_.push_back(Tensor2Wrapper<size_t>::create(tensor));
    }
    {
      Tensor2<uint32_t> tensor;
      buffers_[dev_id]->reserve({1, 1}, &tensor);
      dev_unique_indices_nums_.push_back(tensor);
    }
    {
      Tensor2<uint32_t> tensor;
      host_buffers_[dev_id]->reserve({1, 1}, &tensor);
      host_unique_indices_nums_.push_back(tensor);
    }
    {
      size_t size = 0;
      InclusiveSum((void*)nullptr, size, (uint32_t*)nullptr, (uint32_t*)nullptr,
                   global_batch_size_ * max_feature_num_);
      Tensor2<void> tensor;
      buffers_[dev_id]->reserve({size}, &tensor);
      temp_storage_sum_tensors_.push_back(tensor);
    }
    {
      size_t size = 0;
      SortPairs((void*)nullptr, size, (int64_t*)nullptr, (int64_t*)nullptr, (int64_t*)nullptr,
                (int64_t*)nullptr, global_batch_size_ * max_feature_num_);
      Tensor2<void> tensor;
      buffers_[dev_id]->reserve({size}, &tensor);
      temp_storage_sort_tensors_.push_back(tensor);
    }
  }  // for dev in local_gpu_count
}

std::shared_ptr<GradUpdatePreparer> GradUpdatePreparer::create(
    const size_t global_batch_size, const size_t max_feature_num,
    const size_t max_vocabulary_size_per_gpu,
    std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>& host_buffers,
    std::shared_ptr<ResourcesManager>& resource_mgr) {
  return std::shared_ptr<GradUpdatePreparer>(
      new GradUpdatePreparer(global_batch_size, max_feature_num, max_vocabulary_size_per_gpu,
                             buffers, host_buffers, resource_mgr));
}

void GradUpdatePreparer::operator()(const std::shared_ptr<Tensor>& duplicated_indices,
                                    const size_t local_replica_id) {
  const auto& local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
  const size_t nnz = duplicated_indices->get_num_elements();

  // step1: generate position array
  gen_position_for_indices(duplicated_indices->GetPtrWithType<int64_t>(), nnz,
                           positions_[local_replica_id].get_ptr(), local_gpu->get_stream());

  // step2: sort duplicated_indices
  int32_t end_bit =
      static_cast<int32_t>(std::log2(static_cast<float>(max_vocabulary_size_per_gpu_))) + 1;
  size_t temp_size = temp_storage_sort_tensors_[local_replica_id].get_size_in_bytes();
  CK_CUDA(SortPairs(/*d_temp_storage=*/temp_storage_sort_tensors_[local_replica_id].get_ptr(),
                    /*temp_storage_bytes=*/temp_size,
                    /*d_keys_in=*/duplicated_indices->GetPtrWithType<int64_t>(),
                    /*d_keys_out=*/sort_duplicated_indices_[local_replica_id].get_ptr(),
                    /*d_values_in=*/positions_[local_replica_id].get_ptr(),
                    /*d_values_out=*/sort_positions_[local_replica_id].get_ptr(),
                    /*num_items=*/nnz,
                    /*begin_bit=*/0,
                    /*end_bit=*/end_bit, local_gpu->get_stream(), false));

  // step3: generate indices unique flags
  gen_unique_flags_for_indices(sort_duplicated_indices_[local_replica_id].get_ptr(), nnz,
                               indices_unique_flags_[local_replica_id].get_ptr(),
                               local_gpu->get_stream());

  // step4: generate prefix sum based on indices unique flags
  size_t temp_sum_size = temp_storage_sum_tensors_[local_replica_id].get_size_in_bytes();
  CK_CUDA(InclusiveSum(/*d_temp_storage=*/temp_storage_sum_tensors_[local_replica_id].get_ptr(),
                       /*temp_storage_bytes=*/temp_sum_size,
                       /*d_in=*/indices_unique_flags_[local_replica_id].get_ptr(),
                       /*d_out=*/indices_unique_sums_[local_replica_id].get_ptr(),
                       /*num_items=*/nnz, local_gpu->get_stream()));

  // step5: generate index for unique indices based on unique flags and unique sums
  gen_unique_indexes_for_indices(indices_unique_flags_[local_replica_id].get_ptr(),
                                 indices_unique_sums_[local_replica_id].get_ptr(), nnz,
                                 indices_unique_indexes_[local_replica_id].get_ptr(),
                                 dev_unique_indices_nums_[local_replica_id].get_ptr(),
                                 local_gpu->get_stream());

  // step6: copy dev_unique_indices_num to host
  CK_CUDA(cudaMemcpyAsync(host_unique_indices_nums_[local_replica_id].get_ptr(),
                          dev_unique_indices_nums_[local_replica_id].get_ptr(),
                          sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost, local_gpu->get_stream()));
  resource_mgr_->sync_gpu(local_replica_id);
}

std::shared_ptr<Tensor>& GradUpdatePreparer::get_indices_unique_indexes(
    const size_t local_replica_id) {
  return indices_unique_indexes_interface_[local_replica_id];
}

std::shared_ptr<Tensor>& GradUpdatePreparer::get_sorted_position(const size_t local_replica_id) {
  return sort_positions_interface_[local_replica_id];
}

std::shared_ptr<Tensor>& GradUpdatePreparer::get_sorted_indices(const size_t local_replica_id) {
  return sort_duplicated_indices_interface_[local_replica_id];
}

uint32_t GradUpdatePreparer::get_unique_indices_num(const size_t local_replica_id) {
  return host_unique_indices_nums_[local_replica_id].get_ptr()[0];
}

}  // namespace SparseOperationKit