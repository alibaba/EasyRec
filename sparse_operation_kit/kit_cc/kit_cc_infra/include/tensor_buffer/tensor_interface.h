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

#ifndef TENSOR_INTERFACE_H
#define TENSOR_INTERFACE_H

#include "common.h"
#include <cstddef>

namespace SparseOperationKit {

/*
 * This is the base class used to represent datas.
 */
class Tensor {
 public:
  virtual ~Tensor() {}
  virtual size_t get_size_in_bytes() = 0;
  virtual size_t get_num_elements() = 0;
  virtual bool allocated() const = 0;
  virtual DataType dtype() const = 0;

  template <typename TARGET_TYPE>
  TARGET_TYPE* GetPtrWithType() {
    return reinterpret_cast<TARGET_TYPE*>(get_ptr());
  }

  virtual void* get_ptr() = 0;
};

}  // namespace SparseOperationKit

#endif  // TENSOR_INTERFACE_H