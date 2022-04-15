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

#ifndef CONSTRUCTION_CONTEXT_H
#define CONSTRUCTION_CONTEXT_H

#include "parameters/param_interface.h"
#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"

namespace SparseOperationKit {

class ConstructionContext {
 public:
  virtual ~ConstructionContext(){};
  virtual const std::shared_ptr<ResourcesManager>& get_resource_mgr() const = 0;
  virtual std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>& get_buffer(
      const size_t local_replcia_id) = 0;
  virtual std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>& get_host_buffer(
      const size_t local_replcia_id) = 0;
  virtual size_t get_replica_batch_size() const = 0;
  virtual size_t get_global_batch_size() const = 0;
  virtual size_t get_slot_num() const = 0;
  virtual size_t get_nnz_per_slot() const = 0;
  virtual const std::shared_ptr<ParamInterface>& get_param() const = 0;
  virtual size_t get_max_nnz() const = 0;
  virtual size_t get_max_feature_num() const = 0;
  virtual CombinerType get_combiner() const = 0;
  virtual bool used_for_sparse_embedding() const = 0;
  virtual DataType compute_dtype() const = 0;
  virtual DataType key_dtype() const = 0;
};

using ConstructionContext_t = std::shared_ptr<ConstructionContext>;

class DenseConstructionContext : public ConstructionContext {
 public:
  static std::shared_ptr<DenseConstructionContext> create(
      const std::shared_ptr<ResourcesManager>& resource_mgr,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
          host_buffers,
      const size_t replica_batch_size, const size_t slot_num, const size_t nnz_per_slot,
      const DataType key_dtype, const DataType compute_dtype, 
      std::shared_ptr<ParamInterface> param);

  const std::shared_ptr<ResourcesManager>& get_resource_mgr() const override;
  std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>& get_buffer(
      const size_t local_replcia_id) override;
  std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>& get_host_buffer(
      const size_t local_replcia_id) override;
  size_t get_replica_batch_size() const override;
  size_t get_global_batch_size() const override;
  size_t get_slot_num() const override;
  size_t get_nnz_per_slot() const override;
  const std::shared_ptr<ParamInterface>& get_param() const override;
  size_t get_max_nnz() const override;
  size_t get_max_feature_num() const override;
  CombinerType get_combiner() const override;
  bool used_for_sparse_embedding() const override;
  DataType key_dtype() const override;
  DataType compute_dtype() const override;

 protected:
  DenseConstructionContext(
      const std::shared_ptr<ResourcesManager>& resource_mgr,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
          host_buffers,
      const size_t replica_batch_size, const size_t slot_num, const size_t nnz_per_slot,
      const DataType key_dtype, const DataType compute_dtype, 
      std::shared_ptr<ParamInterface> param);

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>> buffers_;
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>> host_buffers_;
  const size_t replica_batch_size_;
  const size_t global_batch_size_;
  const size_t slot_num_;
  const size_t nnz_per_slot_;
  const DataType key_dtype_;
  const DataType compute_dtype_;
  std::shared_ptr<ParamInterface> param_;
};

using DenseConstructionContext_t = std::shared_ptr<DenseConstructionContext>;

class SparseConstructionContext : public DenseConstructionContext {
 public:
  static std::shared_ptr<SparseConstructionContext> create(
      const std::shared_ptr<ResourcesManager>& resource_mgr,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
          host_buffers,
      const size_t replica_batch_size, const size_t rows_num_per_sample, const size_t max_nnz,
      const size_t max_feature_num, const CombinerType combiner,
      const DataType key_dtype, const DataType compute_dtype, 
      std::shared_ptr<ParamInterface> param);

  size_t get_nnz_per_slot() const override;
  size_t get_max_nnz() const override;
  size_t get_max_feature_num() const override;
  CombinerType get_combiner() const override;
  bool used_for_sparse_embedding() const override;

 private:
  SparseConstructionContext(
      const std::shared_ptr<ResourcesManager>& resource_mgr,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
          host_buffers,
      const size_t replica_batch_size, const size_t rows_num_per_sample, const size_t max_nnz,
      const size_t max_feature_num, const CombinerType combiner, 
      const DataType key_dtype, const DataType compute_dtype, 
      std::shared_ptr<ParamInterface> param);

  const size_t max_nnz_;
  const size_t max_feature_num_;
  const CombinerType combiner_;
};

using SparseConstructionContext_t = std::shared_ptr<SparseConstructionContext>;

}  // namespace SparseOperationKit

#endif  // CONSTRUCTION_CONTEXT_H