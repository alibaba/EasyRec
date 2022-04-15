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

#include "embedding_variable.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

class ReadEmbeddingVariableOp : public OpKernel {
 public:
  explicit ReadEmbeddingVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }
  void Compute(OpKernelContext* ctx) override {
    // TODO: no need to read the resource handle??
    core::RefCountPtr<Var> variable;
    const ResourceHandle& handle = HandleFromInput(ctx, 0);
    auto status = LookupResource(ctx, handle, &variable);

    if (!status.ok()) {  // the first resource handle is not ResourceVariable
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

#ifdef DEBUG
    TensorShape tensor_shape = variable->tensor()->shape();
    std::cout << "tensor shape is: [";
    for (auto iter = tensor_shape.begin(); iter != tensor_shape.end(); ++iter) {
      std::cout << (*iter).size << ",";
    }
    std::cout << "\b]" << std::endl;
#endif
    mutex_lock ml(*variable->mu());
    Tensor* t = variable->tensor();
    OP_REQUIRES_OK(ctx, ctx->set_output("value", *t));
  }

 private:
  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("ReadEmbeddingVariableOp")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("tf_resource"),
                        ReadEmbeddingVariableOp);

}  // namespace tensorflow