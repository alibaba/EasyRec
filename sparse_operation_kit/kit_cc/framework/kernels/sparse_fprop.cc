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
class PluginSparseFpropOp : public AsyncOpKernel {
 public:
  explicit PluginSparseFpropOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
  }
  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {
    Tensor const *global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("global_replica_id", &global_replica_id_tensor), done);
    const int32_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();

    auto work_func = [this, ctx, global_replica_id_value, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = ctx->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      Tensor const *emb_handle_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("emb_handle", &emb_handle_tensor), done);
      Tensor const *values_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("values", &values_tensor), done);
      Tensor const *indices_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("indices", &indices_tensor), done);

      // check input shape
      OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(values_tensor->shape()),
                        errors::Aborted("The shape of values must be 1-D vector"), done);
      OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsVector(indices_tensor->shape()),
                        errors::Aborted("The shape of indices must be 1-D vector"), done);
      OP_REQUIRES_ASYNC(ctx, values_tensor->shape() == indices_tensor->shape(),
                        errors::Aborted("The shape of values and indices must be identical."),
                        done);

      // get output shape for the first time
      if (0 == emb_vector_tensor_shape_.dims()) {
        try {
          SparseOperationKit::Facade::instance()->get_output_shape(emb_handle_tensor,
                                                                   emb_vector_tensor_shape_);
        } catch (std::exception const &error) {
          ctx->SetStatus(errors::Aborted(error.what()));
          done();  // error happens
          return;
        }
      }

      // allocate output
      Tensor *emb_vector_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(0, emb_vector_tensor_shape_, &emb_vector_tensor), done);

      // replica_nnz is a scalar
      Tensor *replica_nnz_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(1, {1}, &replica_nnz_tensor), done);

      // do forward propagation
      try {
        SparseOperationKit::Facade::instance()->forward(emb_handle_tensor, values_tensor,
                                                        indices_tensor, global_replica_id_value,
                                                        training_, emb_vector_tensor,
                                                        replica_nnz_tensor);
      } catch (std::exception const &error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        done();  // error happens
        return;
      }

      done();  // no error
    };

    SOK_TF_SCHE_ASYNC(ctx,
                      SparseOperationKit::Facade::instance()->Schedule(global_replica_id_value,
                                                                       std::move(work_func)),
                      done);
  }

 private:
  bool training_;
  TensorShape emb_vector_tensor_shape_;
};
#else
template <typename Device>
class PluginSparseFpropOp : public OpKernel {
 public:
  explicit PluginSparseFpropOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
  }
  void Compute(OpKernelContext *ctx) override {
    Tensor const *emb_handle_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_handle", &emb_handle_tensor));
    Tensor const *values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
    Tensor const *indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    Tensor const *global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));

    // check input shape
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values_tensor->shape()),
                errors::Aborted("The shape of values must be 1-D vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices_tensor->shape()),
                errors::Aborted("The shape of indices must be 1-D vector"));
    OP_REQUIRES(ctx, values_tensor->shape() == indices_tensor->shape(),
                errors::Aborted("The shape of values and indices must be identical."));

    // get output shape for the first time
    if (0 == emb_vector_tensor_shape_.dims()) {
      try {
        SparseOperationKit::Facade::instance()->get_output_shape(emb_handle_tensor,
                                                                 emb_vector_tensor_shape_);
      } catch (std::exception const &error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        return;
      }
    }

    // allocate output
    Tensor *emb_vector_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape_, &emb_vector_tensor));

    // do forward propagation
    try {
      SparseOperationKit::Facade::instance()->forward(
          emb_handle_tensor, values_tensor, indices_tensor,
          global_replica_id_tensor->scalar<int32_t>()(), training_, emb_vector_tensor);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  bool training_;
  TensorShape emb_vector_tensor_shape_;
};
#endif

REGISTER_KERNEL_BUILDER(Name("PluginSparseFprop")
                            .Device(DEVICE_GPU)
                            .HostMemory("emb_handle")
                            .HostMemory("emb_var_handle")
                            .HostMemory("global_replica_id")
                            .HostMemory("replica_nnz"),
                        PluginSparseFpropOp<GPUDevice>);

}  // namespace tensorflow