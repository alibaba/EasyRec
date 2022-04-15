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

#ifndef RANDOM_UNIFORM_H
#define RANDOM_UNIFORM_H

#include "initializer/initializer_interface.h"

namespace SparseOperationKit {

class RandomUniformInit : public Initializer {
 public:
  static std::shared_ptr<RandomUniformInit> create(const float a, const float b);

  void fill(std::shared_ptr<Tensor> tensor, const size_t sm_count,
            const curandGenerator_t& generator, const cudaStream_t& stream) override;

 private:
  RandomUniformInit(const float a, const float b);

  const float a_, b_;
};

}  // namespace SparseOperationKit

#endif  // RANDOM_UNIFORM_H