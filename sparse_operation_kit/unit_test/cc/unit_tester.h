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

#ifndef INCLUDE_UNIT_TESTER_H
#define INCLUDE_UNIT_TESTER_H

#include <memory>
#include <vector>

#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"
#include "tensorflow/core/framework/tensor.h"

namespace SparseOperationKit {

class UnitTester final {
 public:
  static UnitTester* instance(const std::shared_ptr<ResourcesManager>& resource_mgr);
  void operator delete(void*);

  void test_all_gather_dispatcher(const size_t rows_num_per_sample, const size_t max_nnz,
                                  size_t const global_batch_size, const size_t global_replica_id,
                                  const tensorflow::Tensor* values_tensor,
                                  const tensorflow::Tensor* indices_tensor,
                                  tensorflow::Tensor* values_out_tensor,
                                  tensorflow::Tensor* indices_out_tensor,
                                  tensorflow::Tensor* num_elements_tensor,
                                  tensorflow::Tensor* total_valid_num_tensor);
  void test_csr_conversion_distributed(
      const size_t global_replica_id, const size_t global_batch_size, const size_t slot_num,
      const size_t max_nnz, const tensorflow::Tensor* values_tensor,
      const tensorflow::Tensor* row_indices_tensor, const tensorflow::Tensor* num_elements_tensor,
      const tensorflow::Tensor* total_valid_num_tensor, tensorflow::Tensor* replcia_values_tensor,
      tensorflow::Tensor* replica_csr_row_offsets_tensor, tensorflow::Tensor* replica_nnz_tensor);
  void test_reduce_scatter_dispatcher(const size_t global_replica_id,
                                      const size_t global_batch_size, const size_t slot_num,
                                      const size_t max_nnz, const tensorflow::Tensor* input,
                                      tensorflow::Tensor* output);

 private:
  explicit UnitTester(const std::shared_ptr<ResourcesManager>& resource_mgr);

  std::shared_ptr<ResourcesManager> resource_mgr_;
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>> buffers_;
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>> host_buffers_;

  void try_allocate_memory(const size_t local_replica_id) const;
};

}  // namespace SparseOperationKit

#endif  // INCLUDE_UNIT_TESTER_H