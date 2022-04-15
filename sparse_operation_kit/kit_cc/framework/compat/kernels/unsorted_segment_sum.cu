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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#include "gpu_kernel_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

// UnsortedSegmentSumKernel processes 'input_total_size' elements.
// Each element is mapped from input to output by a combination of its
// 'segment_ids' mapping and 'inner_dim_size'.
template <typename T, typename Index, typename KernelReductionFunctor>
__global__ void UnsortedSegmentCustomKernel(const int64 input_outer_dim_size,
                                            const int64 inner_dim_size,
                                            const int64 output_outer_dim_size,
                                            const Index *__restrict__ segment_ids,
                                            const T *__restrict__ input, T *__restrict__ output) {
  const int64 input_total_size = input_outer_dim_size * inner_dim_size;
  for (int64 input_index : GpuGridRangeX(input_total_size)) {
    const int64 input_segment_index = input_index / inner_dim_size;
    const int64 segment_offset = input_index % inner_dim_size;
    const Index output_segment_index = segment_ids[input_segment_index];
    if (output_segment_index < 0 || output_segment_index >= output_outer_dim_size) {
      continue;
    }
    const int64 output_index = output_segment_index * inner_dim_size + segment_offset;
    KernelReductionFunctor()(output + output_index, __ldg(input + input_index));
  }
}

namespace internal {
// check routines not in the templated class to reduce code size
Status ValidateUnsortedSegmentReduction(OpKernel *op_kernel, OpKernelContext *context,
                                        const Tensor &data, const Tensor &segment_ids,
                                        const Tensor &num_segments) {
  if (!TensorShapeUtils::IsScalar(num_segments.shape())) {
    return errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                   num_segments.shape().DebugString());
  }

  if (!TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape())) {
    return errors::InvalidArgument(
        "data.shape = ", data.shape().DebugString(),
        " does not start with segment_ids.shape = ", segment_ids.shape().DebugString());
  }

  return Status::OK();
}

template <typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC const T SubtleMustCopy(const T &x) {
  static_assert(std::is_integral<T>::value, "SubtleMustCopy can only be used on integer types.");
  auto *to_x = reinterpret_cast<const volatile T *>(&x);
  return *to_x;
}

}  // namespace internal

namespace functor {

template <typename T>
struct Zero {
  EIGEN_STRONG_INLINE T operator()() const { return T(0); }
};

// Atomic reduction functors for the gpu.
template <typename T>
struct AtomicSumOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T *dest, const T &value) {
    GpuAtomicAdd(dest, value);
  }
  static constexpr bool is_associative = std::is_integral<T>::value;
};

template <typename T, typename Index, typename InitialValueF, typename ReductionF>
struct UnsortedSegmentFunctor {
  void operator()(OpKernelContext *ctx, const TensorShape &segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data, typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }
    // Set 'output' to initial value.
    GPUDevice d = ctx->template eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
    TF_CHECK_OK(GpuLaunchKernel(SetToValue<T>, config.block_count, config.thread_per_block, 0,
                                d.stream(), output.size(), output.data(), InitialValueF()()));
    const int64 data_size = data.size();
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }
    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const int64 input_outer_dim_size = segment_ids.dimension(0);
    const int64 input_inner_dim_size = data.dimension(1);
    const int64 output_outer_dim_size = output.dimension(0);
    config = GetGpuLaunchConfig(data_size, d);

    TF_CHECK_OK(GpuLaunchKernel(UnsortedSegmentCustomKernel<T, Index, ReductionF>,
                                config.block_count, config.thread_per_block, 0, d.stream(),
                                input_outer_dim_size, input_inner_dim_size, output_outer_dim_size,
                                segment_ids.data(), data.data(), output.data()));
  }
};
}  // namespace functor

// The UnsortedSegmentReduction OpKernel. The DeviceReductionFunctor
// is the device specific implementation of the reduction. These device
// specific implementations are templated themselves with the corresponding
// initial value functors and reduction functors.
template <typename T, typename Index, typename DeviceReductionFunctor>
class UnsortedSegmentReductionOp : public OpKernel {
 public:
  explicit UnsortedSegmentReductionOp(OpKernelConstruction *context)
      : OpKernel(context), reduction_functor_(DeviceReductionFunctor()) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &data = context->input(0);
    const Tensor &segment_ids = context->input(1);
    const Tensor &num_segments = context->input(2);
    OP_REQUIRES_OK(context, internal::ValidateUnsortedSegmentReduction(this, context, data,
                                                                       segment_ids, num_segments));
    const auto segment_flat = segment_ids.flat<Index>();
    const int64 output_rows = internal::SubtleMustCopy(
        static_cast<int64>(num_segments.dtype() == DT_INT32 ? num_segments.scalar<int32>()()
                                                            : num_segments.scalar<int64>()()));
    OP_REQUIRES(
        context, output_rows >= 0,
        errors::InvalidArgument("Input num_segments == ", output_rows, " must not be negative."));
    TensorShape output_shape;
    output_shape.AddDim(output_rows);
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();
    auto data_flat = data.flat_inner_outer_dims<T, 2>(segment_ids.dims() - 1);
    reduction_functor_(context, segment_ids.shape(), segment_flat, data_flat, output_flat);
  }

 protected:
  DeviceReductionFunctor reduction_functor_;
};

#define REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(type, index_type, initial_value_functor, \
                                            reduction_kernel_functor)                \
  REGISTER_KERNEL_BUILDER(                                                           \
      Name("GPUUnsortedSegmentSum")                                                  \
          .Device(DEVICE_GPU)                                                        \
          .HostMemory("num_segments")                                                \
          .TypeConstraint<type>("T")                                                 \
          .TypeConstraint<index_type>("Tindices"),                                   \
      UnsortedSegmentReductionOp<                                                    \
          type, index_type,                                                          \
          functor::UnsortedSegmentFunctor<type, index_type, initial_value_functor,   \
                                          reduction_kernel_functor>>)

#define REGISTER_SUM_GPU_UNSORTED_KERNELS(type, index_type)                  \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(type, index_type, functor::Zero<type>, \
                                      functor::AtomicSumOpGpu<type>);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, int32)    \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, int64)

TF_CALL_float(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_half(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);

}  // namespace tensorflow
