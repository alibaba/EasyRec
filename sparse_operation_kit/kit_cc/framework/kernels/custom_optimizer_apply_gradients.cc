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

#include "embedding_variable.h"
#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class CustomOptimizerApplyGradientsOp : public OpKernel {
 public:
  explicit CustomOptimizerApplyGradientsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<EmbeddingVariable> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    auto status = LookupResource(ctx, handle, &variable);
    OP_REQUIRES(ctx, status.ok(),
                errors::FailedPrecondition("Error while reading EmbeddingVariable ", handle.name(),
                                           " from container: ", handle.container(),
                                           ". This could mean that you haven't create it. ",
                                           status.ToString()));
    const Tensor* grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    const Tensor* local_indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("local_indices", &local_indices_tensor));
    const Tensor* learning_rate_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("learning_rate", &learning_rate_tensor));
    const Tensor* current_step_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("current_step", &current_step_tensor));

    // check input shape
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(grad_tensor->shape()),
                errors::Aborted("The shape of gradients must be 2-D matrix, "
                                "which is [valid_nums, embedding_vec_size]."));

    try {
      const int local_replica_id =
          SparseOperationKit::GetLocalReplicaIdFromDeviceName(ctx->device()->name());
      SparseOperationKit::Facade::instance()->apply_gradients(
          variable, grad_tensor, local_indices_tensor, local_replica_id,
          learning_rate_tensor->scalar<float>()(), current_step_tensor->scalar<int64_t>()());
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CustomOptimizerApplyGradients")
                            .Device(DEVICE_GPU)
                            .HostMemory("emb_var_handle")
                            .HostMemory("learning_rate")
                            .HostMemory("current_step"),
                        CustomOptimizerApplyGradientsOp<GPUDevice>);

}  // namespace tensorflow