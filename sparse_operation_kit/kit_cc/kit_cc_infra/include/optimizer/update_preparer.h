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

#ifndef UPDATE_PREPARER_H
#define UPDATE_PREPARER_H

#include <memory>

#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {

/**
 * This class is used to do some preparation for
 * applying gradients to params.
 * Each ParamInterface should have one preparer.
 */
class UpdatePreparer {
 public:
  virtual ~UpdatePreparer() {}
  virtual void operator()(const std::shared_ptr<Tensor>& duplicated_indices,
                          const size_t local_replica_id) = 0;
  virtual std::shared_ptr<Tensor>& get_indices_unique_indexes(const size_t local_replica_id) = 0;
  virtual std::shared_ptr<Tensor>& get_sorted_position(const size_t local_replica_id) = 0;
  virtual std::shared_ptr<Tensor>& get_sorted_indices(const size_t local_replica_id) = 0;
  virtual uint32_t get_unique_indices_num(const size_t local_replica_id) = 0;
};

}  // namespace SparseOperationKit

#endif  // UPDATE_PREPARER_H