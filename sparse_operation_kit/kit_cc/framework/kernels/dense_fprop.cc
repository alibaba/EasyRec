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
class PluginDenseFpropOp : public AsyncOpKernel {
 public:
  explicit PluginDenseFpropOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_input", &dynamic_input_));
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

      // check input shape
      if (dynamic_input_) {
        // input values must be a vector
        OP_REQUIRES_ASYNC(
            ctx, TensorShapeUtils::IsVector(values_tensor->shape()),
            errors::Aborted("When dynamic_input is used, the input values must be a 1-D vector."),
            done);
      } else {
        // input values must be [?, slot_num, nnz_per_slot]
        OP_REQUIRES_ASYNC(ctx, 3 == values_tensor->shape().dims(),
                          errors::Aborted("The shape of input values must be 3-D tensor, which is "
                                          "[batchsize, slot_num, nnz_per_slot]."),
                          done);
      }

      // allocate output
      Tensor *emb_vector_tensor = nullptr;
      if (dynamic_input_) {  // the input shape is dynamic
        TensorShape emb_vector_tensor_shape = {values_tensor->NumElements()};
        TensorShape embedding_vec_size_shape;
        try {
          // only embedding_vec_size will bet set in embedding_vec_size_shape.
          SparseOperationKit::Facade::instance()->get_output_shape(emb_handle_tensor,
                                                                   embedding_vec_size_shape,
                                                                   /*dynamic_input=*/true);
        } catch (std::exception const &error) {
          ctx->SetStatus(errors::Aborted(error.what()));
          done();  // if error happens
          return;
        }
        emb_vector_tensor_shape.AppendShape(embedding_vec_size_shape);

        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->allocate_output(0, emb_vector_tensor_shape, &emb_vector_tensor), done);
      } else {  // the input shape is static.
        // get output shape for the first time
        if (0 == emb_vector_tensor_shape_.dims()) {
          try {
            SparseOperationKit::Facade::instance()->get_output_shape(emb_handle_tensor,
                                                                     emb_vector_tensor_shape_);
          } catch (std::exception const &error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            done();  // if error happens
            return;
          }
        }

        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->allocate_output(0, emb_vector_tensor_shape_, &emb_vector_tensor), done);
      }

       // replica_nnz is a scalar
      Tensor *replica_nnz_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(1, {1}, &replica_nnz_tensor), done);

      // do forward propagation
      try {
        SparseOperationKit::Facade::instance()->forward(emb_handle_tensor, values_tensor,
                                                        global_replica_id_value, training_,
                                                        emb_vector_tensor,
                                                        replica_nnz_tensor);
      } catch (std::exception const &error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        done();  // if error happens
        return;
      }

      done();  // no error happens
    };

    SOK_TF_SCHE_ASYNC(ctx,
                      SparseOperationKit::Facade::instance()->Schedule(global_replica_id_value,
                                                                       std::move(work_func)),
                      done);
  }

 private:
  bool training_;
  bool dynamic_input_;
  TensorShape emb_vector_tensor_shape_;
};

#else
template <typename Device>
class PluginDenseFpropOp : public OpKernel {
 public:
  explicit PluginDenseFpropOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_input", &dynamic_input_));
  }
  void Compute(OpKernelContext *ctx) override {
    Tensor const *emb_handle_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_handle", &emb_handle_tensor));
    Tensor const *values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
    Tensor const *global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));

    // check input shape
    if (dynamic_input_) {
      // input values must be a vector
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVector(values_tensor->shape()),
          errors::Aborted("When dynamic_input is used, the input values must be a 1-D vector."));
    } else {
      // input values must be [?, slot_num, nnz_per_slot]
      OP_REQUIRES(ctx, 3 == values_tensor->shape().dims(),
                  errors::Aborted("The shape of input values must be 3-D tensor, which is "
                                  "[batchsize, slot_num, nnz_per_slot]."));
    }

    // allocate output
    Tensor *emb_vector_tensor = nullptr;
    if (dynamic_input_) {  // the input shape is dynamic
      TensorShape emb_vector_tensor_shape = {values_tensor->NumElements()};
      TensorShape embedding_vec_size_shape;
      try {
        // only embedding_vec_size will bet set in embedding_vec_size_shape.
        SparseOperationKit::Facade::instance()->get_output_shape(emb_handle_tensor,
                                                                 embedding_vec_size_shape,
                                                                 /*dynamic_input=*/true);
      } catch (std::exception const &error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        return;
      }
      emb_vector_tensor_shape.AppendShape(embedding_vec_size_shape);

      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape, &emb_vector_tensor));
    } else {  // the input shape is static.
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

      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape_, &emb_vector_tensor));
    }

    // do forward propagation
    try {
      SparseOperationKit::Facade::instance()->forward(emb_handle_tensor, values_tensor,
                                                      global_replica_id_tensor->scalar<int32_t>()(),
                                                      training_, emb_vector_tensor);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  bool training_;
  bool dynamic_input_;
  TensorShape emb_vector_tensor_shape_;
};
#endif

REGISTER_KERNEL_BUILDER(Name("PluginDenseFprop")
                            .Device(DEVICE_GPU)
                            .HostMemory("emb_handle")
                            .HostMemory("emb_var_handle")
                            .HostMemory("global_replica_id")
                            .HostMemory("replica_nnz"),
                        PluginDenseFpropOp<GPUDevice>);

}  // namespace tensorflow