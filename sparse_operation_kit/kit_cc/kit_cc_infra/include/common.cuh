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

#ifndef COMMON_CUH_
#define COMMON_CUH_

#include "common.h"

namespace SparseOperationKit {

template <typename TIN, typename TOUT, typename Lambda>
__global__ void transform_array(const TIN* in, TOUT* out, size_t num_elements, Lambda op) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    out[tid] = op(in[tid]);
  }
}

__global__ void reduce_sum(const size_t* nums, const size_t nums_len, size_t* result);

}  // namespace SparseOperationKit

namespace HugeCTR {

template <typename TOUT, typename TIN>
struct TypeConvertFunc;

template <>
struct TypeConvertFunc<__half, float> {
  static __forceinline__ __device__ __half convert(float val) { return __float2half(val); }
};

template <>
struct TypeConvertFunc<float, __half> {
  static __forceinline__ __device__ float convert(__half val) { return __half2float(val); }
};

template <>
struct TypeConvertFunc<float, float> {
  static __forceinline__ __device__ float convert(float val) { return val; }
};

template <>
struct TypeConvertFunc<float, long long> {
  static __forceinline__ __device__ float convert(long long val) { return static_cast<float>(val); }
};

template <>
struct TypeConvertFunc<float, unsigned int> {
  static __forceinline__ __device__ float convert(unsigned int val) {
    return static_cast<float>(val);
  }
};

}  // namespace HugeCTR

#endif  // COMMON_CUH_