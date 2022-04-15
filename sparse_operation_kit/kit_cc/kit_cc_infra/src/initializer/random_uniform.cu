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

#include "common.cuh"
#include "common.h"
#include "initializer/random_uniform.h"

namespace SparseOperationKit {

RandomUniformInit::RandomUniformInit(const float a, const float b) : a_(a), b_(b) {
  if (a_ >= b_) throw std::runtime_error(ErrorBase + "a must be smaller than b.");
}

std::shared_ptr<RandomUniformInit> RandomUniformInit::create(const float a, const float b) {
  return std::shared_ptr<RandomUniformInit>(new RandomUniformInit(a, b));
}

void RandomUniformInit::fill(std::shared_ptr<Tensor> tensor, const size_t sm_count,
                             const curandGenerator_t& generator, const cudaStream_t& stream) {
  CK_CURAND(curandGenerateUniform(generator, tensor->GetPtrWithType<float>(),
                                  tensor->get_num_elements()));

  float a = a_, b = b_;
  auto op = [a, b] __device__(float val) { return val * (b - a) + a; };
  transform_array<<<sm_count * 2, 1024, 0, stream>>>(tensor->GetPtrWithType<float>(),
                                                     tensor->GetPtrWithType<float>(),
                                                     tensor->get_num_elements(), op);
}

}  // namespace SparseOperationKit