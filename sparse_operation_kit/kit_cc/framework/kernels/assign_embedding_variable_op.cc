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

#define AlreadyInitializedError(ctx, name)                                              \
  do {                                                                                  \
    (ctx)->SetStatus(errors::Aborted((name), " has already been initialized. ",         \
                                     "This might be caused by that sess.run(init_op) ", \
                                     "is called more than once."));                     \
    return;                                                                             \
  } while (0)

template <typename Device>
class AssignEmbeddingVariableOp : public OpKernel {
 public:
  explicit AssignEmbeddingVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("trainable", &trainable_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_hashtable", &use_hashtable_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_and_shape_.dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &dtype_and_shape_.shape));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_dtype", &key_dtype_));
    OP_REQUIRES(ctx, 2 == dtype_and_shape_.shape.dims(), errors::Aborted(
        __FILE__, ":", __LINE__, " ",
        "shape must be [vocabulary_size_per_gpu, embedding_vector_size]."));
    OP_REQUIRES(ctx, dtype_and_shape_.shape.IsFullyDefined(), 
        errors::Aborted(__FILE__, ":", __LINE__, " ", "shape must be fully defined."));
    shape_convertor(ctx);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (initialized_.load(std::memory_order_acquire)) {
      AlreadyInitializedError(ctx, var_name_);
    }
    mutex_lock ml(mu_);
    // check again to see if another thread has initialized it.
    if (initialized_.load(std::memory_order_acquire)) {
      AlreadyInitializedError(ctx, var_name_);
    }

    const Tensor* initial_value_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("initial_value", &initial_value_tensor));
    OP_REQUIRES(
        ctx,
        initial_value_tensor->dtype() == DT_STRING || initial_value_tensor->dtype() == DT_FLOAT,
        errors::Aborted(__FILE__, ":", __LINE__,
                        " Only string or tensor can be used as"
                        " initializer."));
    const Tensor* local_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("local_replica_id", &local_replica_id_tensor));

    std::string variable_name = var_name_;
    // generate unique variable name
    try {
      SparseOperationKit::Facade::instance()->generate_unique_name(trainable_, variable_name);
    } catch (const std::exception& error) {
      ctx->SetStatus(
          errors::Aborted("Error happens when generating unique name, "
                          "due to ",
                          error.what()));
      return;
    }
    OP_REQUIRES(ctx, var_name_ == variable_name,
                errors::Aborted(__FILE__, ":", __LINE__, " there already exist ", var_name_));

    // Create resource for EmbeddingVariable
    core::RefCountPtr<EmbeddingVariable> emb_variable;
    OP_REQUIRES_OK(
        ctx, LookupOrCreateResource<EmbeddingVariable>(ctx, HandleFromInput(ctx, 0), &emb_variable,
                                                       /*creator=*/[](EmbeddingVariable** ptr) {
                                                         *ptr = new EmbeddingVariable();
                                                         return Status::OK();
                                                       }));
    Tensor tensor;  // used to hold the pointer to allocated GPU memory
    try {
      const size_t local_replica_id_value = local_replica_id_tensor->scalar<int32_t>()();
      if (DT_STRING == initial_value_tensor->dtype()) {
        // this will use in-place initializer
        SparseOperationKit::Facade::instance()->create_variables(
            local_replica_id_value, std::string(initial_value_tensor->flat<tstring>()(0)),
            use_hashtable_, dims_, variable_name, trainable_, dtype_and_shape_.dtype,
            key_dtype_, emb_variable, &tensor);
      } else {
        // this will copy initial value to variable's memory
        OP_REQUIRES(ctx, initial_value_tensor->dtype() == dtype_and_shape_.dtype,
                    errors::Aborted("The dtype of initial_value is not compatible "
                                    "with EmbeddingVariable's."));
        OP_REQUIRES(ctx, dtype_and_shape_.dtype == DT_FLOAT,
                    errors::Aborted("The dtype of EmbeddingVariable must be float32."));
        SparseOperationKit::Facade::instance()->create_variables(
            local_replica_id_value, initial_value_tensor, use_hashtable_, 
            dims_, variable_name, trainable_, dtype_and_shape_.dtype, key_dtype_,
            emb_variable, &tensor);
      }
    } catch (const std::exception& error) {
      ctx->SetStatus(
          errors::Aborted(__FILE__, ":", __LINE__, " errors happens due to ", error.what()));
      return;
    }

    // create resource for TF Var
    core::RefCountPtr<Var> tf_variable;
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, HandleFromInput(ctx, 1), &tf_variable,
                                                    /*creator=*/[&tensor](Var** ptr) {
                                                      *ptr = new Var(DT_FLOAT);
                                                      *(*ptr)->tensor() = tensor;
                                                      (*ptr)->is_initialized = true;
                                                      return Status::OK();
                                                    }));

    // set the flag
    initialized_.store(true, std::memory_order_release);
  }

 private:
  bool trainable_;
  DtypeAndPartialTensorShape dtype_and_shape_;
  DataType key_dtype_;
  std::vector<int64_t> dims_;
  bool use_hashtable_;
  std::string var_name_;
  mutex mu_;
  std::atomic<bool> initialized_{false};

  void shape_convertor(OpKernelConstruction* ctx) {
    dims_.clear();
    const auto& shape = dtype_and_shape_.shape;
    for (auto iter = shape.begin(); iter != shape.end(); ++iter) {
      int64_t size_n = (*iter).size;
      if (size_n <= 0) {
        ctx->SetStatus(
            errors::Aborted(__FILE__, ":", __LINE__, " ", "The dim ", size_n, " should be > 0."));
        return;
      }
      dims_.push_back(size_n);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("AssignEmbeddingVariable")
                            .Device(DEVICE_GPU)
                            .HostMemory("emb_var_handle")
                            .HostMemory("tf_var_handle")
                            .HostMemory("local_replica_id"),
                        AssignEmbeddingVariableOp<GPUDevice>);

}  // namespace tensorflow