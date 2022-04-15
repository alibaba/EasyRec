/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SOK_TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
#define SOK_TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include <assert.h>
#include <type_traits>

/**
 * Wrappers and helpers for CUDA device code.
 *
 * Wraps the warp-cooperative intrinsics introduced in CUDA 9 to provide
 * backwards compatibility, see go/volta-porting for details.
 * Provides atomic operations on types that aren't natively supported.
 */

namespace tensorflow {

namespace detail {
// Helper function for atomic accumulation implemented as CAS.
template <typename T, typename F>
__device__ T GpuAtomicCasHelper(T* ptr, F accumulate) {
  T old = *ptr;
  T assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed, accumulate(assumed));
  } while (assumed != old);
  return old;
}

// Overload of above function for half. Note that we don't have
// atomicCAS() for anything less than 32 bits, so we need to include the
// other 16 bits in the operation.
//
// This version is going to be very slow
// under high concurrency, since most threads will be spinning on failing
// their compare-and-swap tests. (The fact that we get false sharing on the
// neighboring fp16 makes this even worse.) If you are doing a large reduction,
// you are much better off with doing the intermediate steps in fp32 and then
// switching to fp16 as late as you can in the calculations.
//
// Note: Assumes little endian.
template <typename F>
__device__ Eigen::half GpuAtomicCasHelper(Eigen::half* ptr, F accumulate) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Not little endian");
#endif
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  assert(!(intptr & 0x1));  // should be 2-aligned.
  if (intptr & 0x2) {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr - 2);
    uint32 result = GpuAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short high = static_cast<unsigned short>(arg >> 16);
      Eigen::half acc = accumulate(Eigen::numext::bit_cast<Eigen::half>(high));
      return (static_cast<uint32>(Eigen::numext::bit_cast<uint16>(acc)) << 16) |
             (arg & 0xffff);
    });
    return Eigen::numext::bit_cast<Eigen::half>(
        static_cast<uint16>(result >> 16));
  } else {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr);
    uint32 result = GpuAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short low = static_cast<unsigned short>(arg & 0xffff);
      Eigen::half acc = accumulate(Eigen::numext::bit_cast<Eigen::half>(low));
      return (arg & 0xffff0000) |
             static_cast<uint32>(Eigen::numext::bit_cast<uint16>(acc));
    });
    return Eigen::numext::bit_cast<Eigen::half>(
        static_cast<uint16>(result & 0xffff));
  }
}


// Helper for range-based for loop using 'delta' increments.
// Usage: see GpuGridRange?() functions below.
template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator &operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator &other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end) : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

template <typename From, typename To>
using ToTypeIfConvertible = typename std::enable_if<std::is_convertible<From, To>::value, To>::type;

template <typename T>
struct CudaSupportedTypeImpl {
  using type = T;
};

template <typename T>
using CudaSupportedType = typename CudaSupportedTypeImpl<T>::type;

template <typename T>
__device__ CudaSupportedType<T> *ToCudaSupportedPtr(T *ptr) {
  return reinterpret_cast<CudaSupportedType<T> *>(ptr);
}

}  // namespace detail

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : GpuGridRangeX(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeX(T count) {
  return detail::GpuGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x,
                                 count);
}

// Helper to set all tensor entries to a specific value.
template <typename T>
__global__ void SetToValue(const int count, T *ptr, T value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : GpuGridRangeX(count)) {
    ptr[i] = value;
  }
}

template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicAdd(T *ptr, U value) {
  return atomicAdd(detail::ToCudaSupportedPtr(ptr), value);
}

__device__ inline Eigen::half GpuAtomicAdd(Eigen::half* ptr,
                                           Eigen::half value) {
  return detail::GpuAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a + value; });
}

}  // namespace tensorflow

#endif  // SOK_TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
