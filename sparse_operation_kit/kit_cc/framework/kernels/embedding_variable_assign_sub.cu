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

#if TF_VERSION_MAJOR == 1

#include <exception>
#include <vector>

#include "common.h"
#include "embedding_variable.h"
#include "tensor_buffer/embedding_buffer.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class EmbeddingVariableAssignSubOp : public OpKernel {
 public:
  explicit EmbeddingVariableAssignSubOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("value", &value_tensor));

    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    auto status = LookupResource(ctx, handle, &variable);

    if (!status.ok()) {  // handle-0 is not Var
      const ResourceHandle& handle = HandleFromInput(ctx, 1);
      auto status = LookupResource(ctx, handle, &variable);
      OP_REQUIRES(ctx, status.ok(),
                  errors::FailedPrecondition(
                      "Error while reading resource variable: ", handle.name(),
                      " from container: ", handle.container(),
                      ", which means the resource handle is neither EmbeddingVariable",
                      " nor ResourceVariable. If you are using TF1, that could also be",
                      " you haven't initialize this Variable, ",
                      "please call sess.run(global_variables_initializer()).", status.ToString()));
    }

    mutex_lock ml(*variable->mu());
    Tensor* var_tensor = variable->tensor();
    OP_REQUIRES(
        ctx, var_tensor->shape().IsSameSize(value_tensor->shape()),
        errors::InvalidArgument("Cannot update EmbeddingVariable with shape ",
                                var_tensor->shape().DebugString(), " using a Tensor with shape ",
                                value_tensor->shape().DebugString(), ", shapes must be equal."));
    OP_REQUIRES(ctx, var_tensor->dtype() == value_tensor->dtype(),
                errors::InvalidArgument("The values's dtype must be the same as that ",
                                        "of EmbeddingVariable's."));
    OP_REQUIRES(ctx, var_tensor->dtype() == DT_FLOAT,
                errors::InvalidArgument("Cannot update EmbeddingVariable with dtype ",
                                        var_tensor->dtype(), ", dtype must be float."));

    // TODO: It's implemented by eigen, could achieve better performance???
    var_tensor->flat<float>().device(ctx->eigen_device<Device>()) -= value_tensor->flat<float>();
  }
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingVariableAssignSub")
                            .Device(DEVICE_GPU)
                            .HostMemory("emb_var_handle")
                            .HostMemory("tf_var_handle"),
                        EmbeddingVariableAssignSubOp<GPUDevice>);

}  // namespace tensorflow

#endif