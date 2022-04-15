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

#include "cc/unit_tester.h"
#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"
#ifdef SOK_ASYNC
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

#ifdef SOK_ASYNC
using ScopedActivateExecutorContext = stream_executor::cuda::ScopedActivateExecutorContext;

class ReduceScatterDispatcherOp : public AsyncOpKernel {
 public:
  explicit ReduceScatterDispatcherOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("global_replica_id", &global_replica_id_tensor), done);
    const size_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();

    auto work_func = [this, ctx, global_replica_id_value, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = ctx->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      const Tensor* input_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input_tensor), done);

      try {
        const auto& resource_mgr = SparseOperationKit::Facade::instance()->get_resource_mgr();
        int64_t replica_batch_size = global_batch_size_ / resource_mgr->get_global_gpu_count();
        int64_t embedding_vec_size = input_tensor->NumElements() / (slot_num_ * global_batch_size_);

        TensorShape shape = {replica_batch_size, slot_num_, embedding_vec_size};

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, shape, &output_tensor), done);

        auto unit_tester = SparseOperationKit::UnitTester::instance(resource_mgr);
        unit_tester->test_reduce_scatter_dispatcher(global_replica_id_value, global_batch_size_,
                                                    slot_num_, max_nnz_, input_tensor,
                                                    output_tensor);

      } catch (const std::exception& error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        done();
        return;
      }
      done();
    };

    SOK_TF_SCHE_ASYNC(ctx,
                      SparseOperationKit::Facade::instance()->Schedule(global_replica_id_value,
                                                                       std::move(work_func)),
                      done);
  }

 private:
  tensorflow::int32 global_batch_size_;
  tensorflow::int32 slot_num_;
  tensorflow::int32 max_nnz_;
};

#else
class ReduceScatterDispatcherOp : public OpKernel {
 public:
  explicit ReduceScatterDispatcherOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const Tensor* input_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    try {
      const auto& resource_mgr = SparseOperationKit::Facade::instance()->get_resource_mgr();
      int64_t replica_batch_size = global_batch_size_ / resource_mgr->get_global_gpu_count();
      int64_t embedding_vec_size = input_tensor->NumElements() / (slot_num_ * global_batch_size_);

      TensorShape shape = {replica_batch_size, slot_num_, embedding_vec_size};

      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output_tensor));

      auto unit_tester = SparseOperationKit::UnitTester::instance(resource_mgr);
      unit_tester->test_reduce_scatter_dispatcher(global_replica_id_tensor->scalar<int32_t>()(),
                                                  global_batch_size_, slot_num_, max_nnz_,
                                                  input_tensor, output_tensor);

    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  tensorflow::int32 global_batch_size_;
  tensorflow::int32 slot_num_;
  tensorflow::int32 max_nnz_;
};
#endif

REGISTER_KERNEL_BUILDER(
    Name("ReduceScatterDispatcher").Device(DEVICE_GPU).HostMemory("global_replica_id"),
    ReduceScatterDispatcherOp);

}  // namespace tensorflow