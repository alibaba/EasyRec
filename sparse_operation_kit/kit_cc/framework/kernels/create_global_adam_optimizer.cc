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
class CreateGlobalAdamOptimizerOp : public OpKernel {
 public:
  explicit CreateGlobalAdamOptimizerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta1", &beta1_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta2", &beta2_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }
  void Compute(OpKernelContext* ctx) override {
    if (!created_.load()) {
      mutex_lock ml(mutex_);
      // check again to see if another thread has created the optimizer handle
      if (!created_.load()) {
        AllocatorAttributes attr;
        attr.set_on_host(true);

        OP_REQUIRES_OK(ctx,
                       ctx->allocate_temp(DT_VARIANT, TensorShape({}), &optimizer_handle_, attr));
        try {
          SparseOperationKit::Facade::instance()->create_optimizer(
              /*optimizer_type=*/"Adam",
              /*optimizer_handle=*/&optimizer_handle_,
              /*hyper_params=*/{{"beta1", beta1_}, {"beta2", beta2_}, {"epsilon", epsilon_}});
        } catch (const std::exception& error) {
          ctx->SetStatus(errors::Aborted(error.what()));
          return;
        }

        created_.store(true);
      }  // second checking
    }    // first checking
    ctx->set_output(0, optimizer_handle_);
  }

 private:
  float beta1_;
  float beta2_;
  float epsilon_;
  Tensor optimizer_handle_;
  std::atomic<bool> created_{false};
  mutex mutex_;
};

REGISTER_KERNEL_BUILDER(
    Name("CreateGlobalAdamOptimizer").Device(DEVICE_GPU).HostMemory("optimizer_handle"),
    CreateGlobalAdamOptimizerOp<GPUDevice>);

}  // namespace tensorflow