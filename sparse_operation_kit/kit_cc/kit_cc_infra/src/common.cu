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

#include <assert.h>

#include "common.cuh"

namespace SparseOperationKit {

// TODO: optimize this function
__global__ void reduce_sum(const size_t* nums, const size_t nums_len, size_t* result) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (0 == gid && result != nullptr) {
    result[0] = 0;
    for (size_t j = 0; j < nums_len; j++) result[0] += nums[j];
  }
}

/*check the numerics is Inf or Nan*/
template <typename T>
__global__ void check_numerics_kernel(const T* data, uint32_t size) {
#if NDEBUG
  return;
#else
  const uint32_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_threads = blockDim.x * gridDim.x;
  for (uint32_t tid = tid_base; tid < size; tid += num_threads) {
    assert(!isnan(data[tid]) && "error: check_numerics faild, got Nan.");
    assert(!isinf(data[tid]) && "error: check_numerics faild, got Inf.");
  }
#endif
}

template <typename T>
void check_numerics(const T* data, uint32_t size, cudaStream_t& stream) {
  const size_t block_size = 1024;
  const size_t grid_size = (size + block_size - 1) / block_size;
  check_numerics_kernel<<<grid_size, block_size, 0, stream>>>(data, size);
}

template void check_numerics(const float* data, uint32_t size, cudaStream_t& stream);

}  // namespace SparseOperationKit