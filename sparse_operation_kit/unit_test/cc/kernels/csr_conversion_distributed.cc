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

class CsrConversionDistribtuedOp : public OpKernel {
 public:
  explicit CsrConversionDistribtuedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const Tensor* valeus_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &valeus_tensor));
    const Tensor* row_indcies_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_indices", &row_indcies_tensor));
    const Tensor* total_valid_num_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("total_valid_num", &total_valid_num_tensor));

    Tensor* replica_values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {global_batch_size_ * slot_num_ * max_nnz_},
                                             &replica_values_tensor));
    Tensor* replica_csr_row_offsets_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {global_batch_size_ * slot_num_ + 1},
                                             &replica_csr_row_offsets_tensor));
    Tensor* replica_nnz_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {1}, &replica_nnz_tensor));

    try {
      const auto& resource_mgr = SparseOperationKit::Facade::instance()->get_resource_mgr();
      auto unit_tester = SparseOperationKit::UnitTester::instance(resource_mgr);

      unit_tester->test_csr_conversion_distributed(
          global_replica_id_tensor->scalar<int32_t>()(), static_cast<size_t>(global_batch_size_),
          static_cast<size_t>(slot_num_), static_cast<size_t>(max_nnz_), valeus_tensor,
          row_indcies_tensor, nullptr, total_valid_num_tensor, replica_values_tensor,
          replica_csr_row_offsets_tensor, replica_nnz_tensor);
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  // FIXME: in TF 2.4, int64_t is not equal to long long int
  tensorflow::int64 global_batch_size_;
  tensorflow::int64 slot_num_;
  tensorflow::int64 max_nnz_;
};

REGISTER_KERNEL_BUILDER(Name("CsrConversionDistributed")
                            .Device(DEVICE_GPU)
                            .HostMemory("global_replica_id")
                            .HostMemory("total_valid_num")
                            .HostMemory("replica_nnz"),
                        CsrConversionDistribtuedOp);

}  // namespace tensorflow