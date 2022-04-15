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

#include <exception>
#include <vector>

#include "embedding_variable.h"
#include "facade.h"
#include "tensor_buffer/embedding_buffer.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class CreateVarOp : public OpKernel {
 public:
  explicit CreateVarOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_and_shape_.dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &dtype_and_shape_.shape));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }

  void Compute(OpKernelContext *ctx) override {
    if (var_name_ == ResourceHandle::ANONYMOUS_NAME) {
      AllocatorAttributes attr;
      attr.set_on_host(true);

      // create handle for EmbeddingVariable
      Tensor embedding_variable_handle;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &embedding_variable_handle, attr));
      embedding_variable_handle.scalar<ResourceHandle>()() = MakeResourceHandle<EmbeddingVariable>(
          ctx,
          /*container=*/"EmbeddingVariableContainer",
          /*name=*/var_name_, std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
      ctx->set_output(0, embedding_variable_handle);

      // create handle for TF Var
      Tensor tf_variable_handle;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &tf_variable_handle, attr));
      tf_variable_handle.scalar<ResourceHandle>()() =
          MakeResourceHandle<Var>(ctx, /*container=*/"VariableContainer", /*name=*/var_name_,
                                  std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
      ctx->set_output(1, tf_variable_handle);
    } else {
      if (!initialized_.load(std::memory_order_acquire)) {
        mutex_lock ml(mutex_);
        // Checking again to see if another thread has initialized the resource.
        if (!initialized_.load(std::memory_order_acquire)) {
          AllocatorAttributes attr;
          attr.set_on_host(true);

          // create handle for EmbeddingVariable
          OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                                 &embedding_variable_handle_, attr));
          embedding_variable_handle_.scalar<ResourceHandle>()() =
              MakeResourceHandle<EmbeddingVariable>(
                  ctx, /*container=*/"EmbeddingVariableContainer",
                  /*name=*/var_name_, std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});

          // create handle for TF Var
          OP_REQUIRES_OK(
              ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &tf_variable_handle_, attr));
          tf_variable_handle_.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
              ctx, /*container=*/"VariableContainer",
              /*name=*/var_name_, std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});

          initialized_.store(true, std::memory_order_release);
        }
      }
      ctx->set_output(0, embedding_variable_handle_);
      ctx->set_output(1, tf_variable_handle_);
    }
  }

 private:
  std::string var_name_;
  mutex mutex_;
  Tensor embedding_variable_handle_;
  Tensor tf_variable_handle_;
  std::atomic<bool> initialized_{false};
  DtypeAndPartialTensorShape dtype_and_shape_;
};

REGISTER_KERNEL_BUILDER(
    Name("CreateVar").Device(DEVICE_GPU).HostMemory("emb_var_handle").HostMemory("tf_var_handle"),
    CreateVarOp<GPUDevice>);

}  // namespace tensorflow