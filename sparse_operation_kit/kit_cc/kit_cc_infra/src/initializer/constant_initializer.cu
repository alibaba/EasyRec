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
#include "initializer/constant_initializer.h"

namespace SparseOperationKit {

ConstantInit::ConstantInit(const float value) : value_(value) {}

std::shared_ptr<ConstantInit> ConstantInit::create(const float value) {
  return std::shared_ptr<ConstantInit>(new ConstantInit(value));
}

void ConstantInit::fill(std::shared_ptr<Tensor> tensor, const size_t sm_count,
                        const curandGenerator_t& generator, const cudaStream_t& stream) {
  float value = value_;
  auto op = [value] __device__(float val) { return value; };
  transform_array<<<sm_count * 2, 1024, 0, stream>>>(tensor->GetPtrWithType<float>(),
                                                     tensor->GetPtrWithType<float>(),
                                                     tensor->get_num_elements(), op);
}

}  // namespace SparseOperationKit