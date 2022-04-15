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
#include "tensorflow/core/framework/resource_var.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class DumpToFileOp : public OpKernel {
 public:
  explicit DumpToFileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<EmbeddingVariable> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    auto status = LookupResource(ctx, handle, &variable);
    OP_REQUIRES(ctx, status.ok(),
                errors::FailedPrecondition("Error in reading EmbeddingVariable: ", handle.name(),
                                           " from container: ", handle.container(),
                                           ". This coule mean that you haven't create it. ",
                                           status.ToString()));

    const Tensor* filepath_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("filepath", &filepath_tensor));

    Tensor* status_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &status_tensor));

    try {
      SparseOperationKit::Facade::instance()->dump_to_file(variable,
                                                           filepath_tensor->flat<tstring>()(0));
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
    status_tensor->flat<tstring>()(0) = "saved.";
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DumpToFile").Device(DEVICE_GPU).HostMemory("var_handle").HostMemory("filepath"),
    DumpToFileOp<GPUDevice>);

}  // namespace tensorflow