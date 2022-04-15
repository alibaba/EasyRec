/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_

#include "gpu_device_functions.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#define GPU_1D_KERNEL_LOOP(i, n) for (int i : ::tensorflow::GpuGridRangeX<int>(n))

namespace tensorflow {

using gpuStream_t = cudaStream_t;

// Returns a raw reference to the current cuda stream. Required by a
// number of kernel calls (for which StreamInterface* does not work),
// i.e. CUB and certain cublas primitives.
inline const gpuStream_t &GetGpuStream(OpKernelContext *context) {
  const gpuStream_t *ptr = CHECK_NOTNULL(reinterpret_cast<const gpuStream_t *>(
      context->op_device_context()->stream()->implementation()->GpuStreamMemberHack()));
  return *ptr;
}

// Launches a GPU kernel through cudaLaunchKernel in CUDA environment, or
// hipLaunchKernel in ROCm environment with the given arguments.
//
// The kernel parameters 'Ts' must be constructible from the arguments 'Args'.
template <typename... Ts, typename... Args>
Status GpuLaunchKernel(void (*function)(Ts...), dim3 grid_dim, dim3 block_dim,
                       size_t shared_memory_size_bytes, gpuStream_t stream, Args... arguments) {
  static_assert(detail::NoneIsReference<Ts...>(),
                "Kernels with reference arguments have undefined behaviour.");
  auto func_ptr = absl::bit_cast<const void *>(function);
  // Cast arguments and forward them as an array of pointers.
  auto args_tuple = std::tuple<Ts...>(arguments...);
  auto arg_ptrs = detail::GetArrayOfElementPointers(&args_tuple);
  auto result = cudaLaunchKernel(func_ptr, grid_dim, block_dim, arg_ptrs.data(),
                                 shared_memory_size_bytes, stream);
  if (result != cudaSuccess) {
    return errors::Internal(cudaGetErrorString(result));
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
