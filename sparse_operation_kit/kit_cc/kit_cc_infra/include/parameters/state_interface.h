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

#ifndef STATE_INTERFACE_H
#define STATE_INTERFACE_H

#include <memory>

#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {

/**
 * This class only contains serveral tensors for a Variable.
 * And the number of the tensors is equal to the number of
 * GPU.
 */
class States {
 public:
  virtual ~States() {}
  virtual void init(const size_t global_replica_id) = 0;
  virtual std::shared_ptr<Tensor>& get_tensor(const size_t local_replica_id) = 0;
};

}  // namespace SparseOperationKit

#endif  // STATE_INTERFACE_H