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

#include "operation/construction_context.h"

namespace SparseOperationKit {

DenseConstructionContext::DenseConstructionContext(
    const std::shared_ptr<ResourcesManager>& resource_mgr,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
        host_buffers,
    const size_t replica_batch_size, const size_t slot_num, const size_t nnz_per_slot,
    const DataType key_dtype, const DataType compute_dtype, 
    std::shared_ptr<ParamInterface> param)
    : resource_mgr_(resource_mgr),
      buffers_(buffers),
      host_buffers_(host_buffers),
      replica_batch_size_(replica_batch_size),
      global_batch_size_(replica_batch_size_ * resource_mgr_->get_global_gpu_count()),
      slot_num_(slot_num),
      nnz_per_slot_(nnz_per_slot),
      key_dtype_(key_dtype),
      compute_dtype_(compute_dtype),
      param_(param) {}

std::shared_ptr<DenseConstructionContext> DenseConstructionContext::create(
    const std::shared_ptr<ResourcesManager>& resource_mgr,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
        host_buffers,
    const size_t replica_batch_size, const size_t slot_num, const size_t nnz_per_slot,
    const DataType key_dtype, const DataType compute_dtype, 
    std::shared_ptr<ParamInterface> param) {
  return std::shared_ptr<DenseConstructionContext>(new DenseConstructionContext(
      resource_mgr, buffers, host_buffers, replica_batch_size, 
      slot_num, nnz_per_slot, key_dtype, compute_dtype, param));
}

const std::shared_ptr<ResourcesManager>& DenseConstructionContext::get_resource_mgr() const {
  return resource_mgr_;
}

std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>&
DenseConstructionContext::get_buffer(const size_t local_replcia_id) {
  if (local_replcia_id >= buffers_.size())
    throw std::runtime_error(ErrorBase + "local_replcia_id is out of the range of buffers' size.");

  return buffers_[local_replcia_id];
}

std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>&
DenseConstructionContext::get_host_buffer(const size_t local_replcia_id) {
  if (local_replcia_id >= host_buffers_.size())
    throw std::runtime_error(ErrorBase +
                             "local_replcia_id is out of the range of host buffers' size.");

  return host_buffers_[local_replcia_id];
}

size_t DenseConstructionContext::get_replica_batch_size() const { return replica_batch_size_; }

size_t DenseConstructionContext::get_global_batch_size() const { return global_batch_size_; }

size_t DenseConstructionContext::get_slot_num() const { return slot_num_; }

size_t DenseConstructionContext::get_nnz_per_slot() const { return nnz_per_slot_; }

const std::shared_ptr<ParamInterface>& DenseConstructionContext::get_param() const {
  return param_;
}

size_t DenseConstructionContext::get_max_nnz() const {
  throw std::runtime_error(ErrorBase + "DenseConstructionContext does not have max_nnz attribute.");
}

size_t DenseConstructionContext::get_max_feature_num() const {
  throw std::runtime_error(ErrorBase +
                           "DenseConstructionContext does not have max_feature_num attribute.");
}

CombinerType DenseConstructionContext::get_combiner() const {
  throw std::runtime_error(ErrorBase +
                           "DenseConstructionContext does not have combiner attribute.");
}

bool DenseConstructionContext::used_for_sparse_embedding() const { return false; }

DataType DenseConstructionContext::key_dtype() const { return key_dtype_; }

DataType DenseConstructionContext::compute_dtype() const { return compute_dtype_; }

SparseConstructionContext::SparseConstructionContext(
    const std::shared_ptr<ResourcesManager>& resource_mgr,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
        host_buffers,
    const size_t replica_batch_size, const size_t rows_num_per_sample, const size_t max_nnz,
    const size_t max_feature_num, const CombinerType combiner,
    const DataType key_dtype, const DataType compute_dtype, 
    std::shared_ptr<ParamInterface> param)
    : DenseConstructionContext(resource_mgr, buffers, host_buffers, replica_batch_size,
                               /*slot_num=*/rows_num_per_sample,
                               /*nnz_per_slot=*/0, key_dtype, compute_dtype, param),
      max_nnz_(max_nnz),
      max_feature_num_(max_feature_num),
      combiner_(combiner) {}

std::shared_ptr<SparseConstructionContext> SparseConstructionContext::create(
    const std::shared_ptr<ResourcesManager>& resource_mgr,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
        host_buffers,
    const size_t replica_batch_size, const size_t rows_num_per_sample, const size_t max_nnz,
    const size_t max_feature_num, const CombinerType combiner,
    const DataType key_dtype, const DataType compute_dtype, 
    std::shared_ptr<ParamInterface> param) {
  return std::shared_ptr<SparseConstructionContext>(new SparseConstructionContext(
      resource_mgr, buffers, host_buffers, replica_batch_size, rows_num_per_sample, max_nnz,
      max_feature_num, combiner, key_dtype, compute_dtype, param));
}

size_t SparseConstructionContext::get_nnz_per_slot() const {
  throw std::runtime_error(ErrorBase +
                           "SparseConstructionContext does not have nnz_per_slot attribute.");
}

size_t SparseConstructionContext::get_max_nnz() const { return max_nnz_; }

size_t SparseConstructionContext::get_max_feature_num() const { return max_feature_num_; }

CombinerType SparseConstructionContext::get_combiner() const { return combiner_; }

bool SparseConstructionContext::used_for_sparse_embedding() const { return true; }

}  // namespace SparseOperationKit