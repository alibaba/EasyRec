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

#ifndef GRAD_UPDATE_PREPARER_H
#define GRAD_UPDATE_PREPARER_H

#include <memory>

#include "optimizer/update_preparer.h"
#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"
#include "tensor_buffer/tensor2.hpp"

namespace SparseOperationKit {

/**
 * This class is used by optimizers except AtomicSGD.
 */
class GradUpdatePreparer : public UpdatePreparer {
  template <typename T>
  using Tensor2 = HugeCTR::Tensor2<T>;
  template <typename T>
  using Tensors2 = HugeCTR::Tensors2<T>;

 public:
  static std::shared_ptr<GradUpdatePreparer> create(
      const size_t global_batch_size, const size_t max_feature_num,
      const size_t max_vocabulary_size_per_gpu,
      std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
          host_buffers,
      std::shared_ptr<ResourcesManager>& resource_mgr);

  void operator()(const std::shared_ptr<Tensor>& duplicated_indices,
                  const size_t local_replica_id) override;
  std::shared_ptr<Tensor>& get_indices_unique_indexes(const size_t local_replica_id) override;
  std::shared_ptr<Tensor>& get_sorted_position(const size_t local_replica_id) override;
  std::shared_ptr<Tensor>& get_sorted_indices(const size_t local_replica_id) override;
  uint32_t get_unique_indices_num(const size_t local_replica_id) override;

 private:
  GradUpdatePreparer(
      const size_t global_batch_size, const size_t max_feature_num,
      const size_t max_vocabulary_size_per_gpu,
      std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>>&
          host_buffers,
      std::shared_ptr<ResourcesManager>& resource_mgr);

  const size_t global_batch_size_;
  const size_t max_feature_num_;
  const size_t max_vocabulary_size_per_gpu_;

  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>> buffers_;
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>> host_buffers_;
  std::shared_ptr<ResourcesManager> resource_mgr_;

  Tensors2<int64_t> positions_;       // store the indexes of original duplicated_indices
  Tensors2<int64_t> sort_positions_;  // store the sorted indexes of original duplicated_indices
  std::vector<std::shared_ptr<Tensor>> sort_positions_interface_;
  Tensors2<int64_t>
      sort_duplicated_indices_;  // store the sorted duplicated_indices TODO: make it template
  std::vector<std::shared_ptr<Tensor>> sort_duplicated_indices_interface_;
  Tensors2<uint32_t> indices_unique_flags_;  // store the flags of whether this indices is unique
  Tensors2<uint32_t> indices_unique_sums_;   // store the prefix sum of unique indices
  Tensors2<size_t> indices_unique_indexes_;  // store the sorted index of unique indices
  std::vector<std::shared_ptr<Tensor>> indices_unique_indexes_interface_;
  Tensors2<uint32_t> dev_unique_indices_nums_;
  Tensors2<uint32_t> host_unique_indices_nums_;
  Tensors2<void> temp_storage_sort_tensors_;  // temp spaces for SortPairs
  Tensors2<void> temp_storage_sum_tensors_;   // temp spaces for InclusiveSum
};

}  // namespace SparseOperationKit

#endif  // GRAD_UPDATE_PREPARER_H