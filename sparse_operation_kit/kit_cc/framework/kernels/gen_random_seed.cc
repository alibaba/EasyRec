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

#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class GenRandomSeedOp : public OpKernel {
 public:
  explicit GenRandomSeedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
  }
  void Compute(OpKernelContext* ctx) override {
    Tensor* global_seed_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &global_seed_tensor));

    try {
      SparseOperationKit::Facade::instance()->get_random_seed(reinterpret_cast<uint64_t*>(&seed_));
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }

    global_seed_tensor->scalar<int64_t>()(0) = seed_;
  }

 private:
  tensorflow::int64 seed_;
};

REGISTER_KERNEL_BUILDER(Name("GenRandomSeed").Device(DEVICE_GPU).HostMemory("global_seed"),
                        GenRandomSeedOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("GenRandomSeed").Device(DEVICE_CPU).HostMemory("global_seed"),
                        GenRandomSeedOp<CPUDevice>);

}  // namespace tensorflow