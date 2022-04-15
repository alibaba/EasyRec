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

#ifndef CONSTANT_INITIALIZER_H
#define CONSTANT_INITIALIZER_H

#include "initializer/initializer_interface.h"

namespace SparseOperationKit {

class ConstantInit : public Initializer {
 public:
  static std::shared_ptr<ConstantInit> create(const float value);

  void fill(std::shared_ptr<Tensor> tensor, const size_t sm_count,
            const curandGenerator_t& generator, const cudaStream_t& stream) override;

 private:
  ConstantInit(const float value);

  const float value_;
};

}  // namespace SparseOperationKit

#endif  // CONSTANT_INITIALIZER_H