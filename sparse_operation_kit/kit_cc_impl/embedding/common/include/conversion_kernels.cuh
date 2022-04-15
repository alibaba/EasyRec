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

#ifndef CONVERSION_KERNELS_CUH_
#define CONVERSION_KERNELS_CUH_

namespace SparseOperationKit {

template <typename Func, typename TIN>
__global__ void boolean_vector(const TIN* input, const size_t elem_size, Func fn,
                               bool* boolean_out) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid = blockDim.x * gridDim.x;
  for (size_t i = gid; i < elem_size; i += grid) {
    boolean_out[i] = fn(input[i]);
  }
}

}  // namespace SparseOperationKit

#endif  // CONVERSION_KERNELS_CUH_