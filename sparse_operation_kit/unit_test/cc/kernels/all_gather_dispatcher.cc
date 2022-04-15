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

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

class AllGatherDispatcherOp : public OpKernel {
 public:
  explicit AllGatherDispatcherOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rows_num_per_sample", &rows_num_per_sample_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
    const Tensor* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    const Tensor* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const Tensor* num_replicas_in_sync_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("num_replicas_in_sync", &num_replicas_in_sync_tensor));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values_tensor->shape()),
                errors::Aborted("values must be vector."));
    OP_REQUIRES(ctx, values_tensor->shape() == indices_tensor->shape(),
                errors::Aborted("indices.shape must be equal to values.shape."));

    Tensor* values_out_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {global_batch_size_ * rows_num_per_sample_ * max_nnz_},
                                        &values_out_tensor));
    Tensor* indices_out_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, {global_batch_size_ * rows_num_per_sample_ * max_nnz_},
                                        &indices_out_tensor));
    Tensor* num_elements_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {num_replicas_in_sync_tensor->scalar<int32_t>()()},
                                             &num_elements_tensor));
    Tensor* total_valid_num_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, {1}, &total_valid_num_tensor));

    try {
      const auto& resource_mgr = SparseOperationKit::Facade::instance()->get_resource_mgr();
      auto unit_tester = SparseOperationKit::UnitTester::instance(resource_mgr);
      unit_tester->test_all_gather_dispatcher(
          static_cast<size_t>(rows_num_per_sample_), static_cast<size_t>(max_nnz_),
          static_cast<size_t>(global_batch_size_), global_replica_id_tensor->scalar<int32_t>()(),
          values_tensor, indices_tensor, values_out_tensor, indices_out_tensor, num_elements_tensor,
          total_valid_num_tensor);
    } catch (std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  // FIXME: in TF 2.4, int64_t is not equal to long long int
  tensorflow::int64 global_batch_size_;
  tensorflow::int64 rows_num_per_sample_;
  tensorflow::int64 max_nnz_;
};

REGISTER_KERNEL_BUILDER(Name("AllGatherDispatcher")
                            .Device(DEVICE_GPU)
                            .HostMemory("global_replica_id")
                            .HostMemory("num_replicas_in_sync"),
                        AllGatherDispatcherOp);

}  // namespace tensorflow