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

#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"
// #if defined(SOK_ASYNC) && defined(ASYNC_OP)
#ifdef SOK_ASYNC
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif
#include <exception>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

// #if defined(SOK_ASYNC) && defined(ASYNC_OP)
#ifdef SOK_ASYNC
using ScopedActivateExecutorContext = stream_executor::cuda::ScopedActivateExecutorContext;

template <typename Device>
class PluginBpropOp : public AsyncOpKernel {
 public:
  explicit PluginBpropOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    Tensor const* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("global_replica_id", &global_replica_id_tensor), done);
    const int32_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();
    Tensor const* replica_nnz_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("replica_nnz", &replica_nnz_tensor), done);
    const size_t h_replica_nnz = replica_nnz_tensor->scalar<size_t>()();

    auto work_func = [ctx, global_replica_id_value, h_replica_nnz, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = ctx->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      Tensor const* emb_handle_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("emb_handle", &emb_handle_tensor), done);
      Tensor const* top_gradient_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("top_gradient", &top_gradient_tensor), done);

      try {
        // get grad shape
        TensorShape grad_shape{ static_cast<tensorflow::int64>(h_replica_nnz) };
        TensorShape _shape;
        SparseOperationKit::Facade::instance()->get_grad_shape(global_replica_id_value,
                                                               emb_handle_tensor, _shape);
        grad_shape.AppendShape(_shape);
        Tensor* gradient_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, grad_shape, &gradient_tensor), done);
        OP_REQUIRES_ASYNC(ctx, static_cast<tensorflow::int64>(h_replica_nnz) == gradient_tensor->dim_size(0), 
                          errors::Aborted(__FILE__, ":", __LINE__, " ",
                          "h_replica_nnz from forward is not consistent with gradient shape."), done);
        Tensor* value_index_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->allocate_output(1, {static_cast<tensorflow::int64>(h_replica_nnz)}, &value_index_tensor),
            done);

        // do backward propagation
        SparseOperationKit::Facade::instance()->backward(emb_handle_tensor, global_replica_id_value,
                                                         top_gradient_tensor, gradient_tensor,
                                                         value_index_tensor);
      } catch (std::exception const& error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        done();  // if errors happens, let parent thread know.
        return;
      }
      done();  // no error happens
    };

    SOK_TF_SCHE_ASYNC(ctx,
                      SparseOperationKit::Facade::instance()->Schedule(global_replica_id_value,
                                                                       std::move(work_func)),
                      done);
  }
};
#else
template <typename Device>
class PluginBpropOp : public OpKernel {
 public:
  explicit PluginBpropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor const* emb_handle_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_handle", &emb_handle_tensor));
    Tensor const* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    Tensor const* top_gradient_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_gradient", &top_gradient_tensor));

    // In TensorFlow, the gradient for GatherOp is IndexedSlices tensor.
    // So we would better return such tensor to tensorflow.
    // For IndexedSlices, it contains:
    // values: A tensor of any dtype with [D0, D1, ..., Dn]
    // indices: A 1-D integer tensor with shape [D0]
    // dense_shape: A 1-D integer tensor containing the shape of the corresponding dense tensor.

    try {
      // get grad shape
      TensorShape grad_shape;
      SparseOperationKit::Facade::instance()->get_grad_shape(
          global_replica_id_tensor->scalar<int32_t>()(), emb_handle_tensor, grad_shape);
      Tensor* gradient_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_shape, &gradient_tensor));
      Tensor* value_index_tensor = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(1, {gradient_tensor->dim_size(0)}, &value_index_tensor));

      // do backward propagation
      SparseOperationKit::Facade::instance()->backward(
          emb_handle_tensor, global_replica_id_tensor->scalar<int32_t>()(), top_gradient_tensor,
          gradient_tensor, value_index_tensor);
    } catch (std::exception const& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};
#endif

REGISTER_KERNEL_BUILDER(
    Name("PluginBprop")
      .Device(DEVICE_GPU)
      .HostMemory("emb_handle")
      .HostMemory("global_replica_id")
      .HostMemory("replica_nnz"),
    PluginBpropOp<GPUDevice>);

}  // namespace tensorflow