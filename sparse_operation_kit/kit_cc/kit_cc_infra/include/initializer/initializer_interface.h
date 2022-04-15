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

#ifndef INITIALIZER_INTERFACE_H
#define INITIALIZER_INTERFACE_H

#include <cuda_runtime.h>
#include <curand.h>

#include <memory>
#include <string>

#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {

enum class InitializerType { RandomUniform, Ones, Zeros };

class Initializer {
 public:
  virtual ~Initializer() {}
  virtual void fill(std::shared_ptr<Tensor> tensor, const size_t sm_count,
                    const curandGenerator_t& generator, const cudaStream_t& stream) = 0;

  static std::shared_ptr<Initializer> Get(const std::string initializer);
};

}  // namespace SparseOperationKit

#endif  // INITIALIZER_INTERFACE_H