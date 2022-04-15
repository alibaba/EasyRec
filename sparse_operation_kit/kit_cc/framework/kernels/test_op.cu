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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/stream.h"

#ifdef SOK_ASYNC
// these headers are only needed in AsyncOpKernel
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif

#include <mpi.h>

#include <chrono>
#include <mutex>
#include <thread>
#include <type_traits>
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

#define CK_MPI(ctx, cmd)                                                                          \
  do {                                                                                            \
    auto retval = (cmd);                                                                          \
    if (MPI_SUCCESS != retval) {                                                                  \
      (ctx)->SetStatus(                                                                           \
          errors::Aborted(__FILE__, ":", __LINE__, ": MPI error code ", std::to_string(retval))); \
      return;                                                                                     \
    }                                                                                             \
  } while (0)

#define CK_MPI_ASYNC(ctx, cmd, done) \
  do {                               \
    _CK_MPI(ctx, cmd);               \
    (done)();                        \
  } while (0)

namespace SparseOperationKit {
namespace {

template <typename Type>
__global__ void TestOpCudaKernel(Type const* input_ptr, Type* output_ptr, const uint32_t elem_num) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t grid = blockDim.x * gridDim.x;
  for (uint32_t i = gid; i < elem_num; i += grid) {
    output_ptr[i] = input_ptr[i];
  }
}

}  // anonymous namespace
}  // namespace SparseOperationKit

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

#ifdef SOK_ASYNC
using ScopedActivateExecutorContext = stream_executor::cuda::ScopedActivateExecutorContext;

template <typename Device>
class TestOp : public AsyncOpKernel {
 public:
  explicit TestOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx), mu_() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unique_op_name", &unique_op_name_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto work_func = [this, ctx, done]() {
      if (std::is_same<Device, CPUDevice>::value) {
        // did no thing
      } else if (std::is_same<Device, GPUDevice>::value) {
        // Ensure that within the callback, the proper GPU settings are
        // configured.
        auto stream = ctx->op_device_context()->stream();
        ScopedActivateExecutorContext scoped_activation{stream->parent()};
      } else {
        ctx->SetStatus(errors::Aborted("Not supported Device Type"));
        done();
        return;
      }

      Tensor const* input_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("x", &input_tensor), done);
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor),
                           done);

      const cudaStream_t stream =
          stream_executor::gpu::AsGpuStreamValue(ctx->op_device_context()->stream());
      // constexpr size_t blocks = 256ul;
      // const size_t grids = (input_tensor->NumElements() + blocks - 1) / blocks;
      // SparseOperationKit::TestOpCudaKernel<<<grids, blocks, 0, stream>>>(
      //                                 reinterpret_cast<const float*>(input_tensor->data()),
      //                                 reinterpret_cast<float*>(output_tensor->data()),
      //                                 input_tensor->NumElements());
#ifdef USE_NVTX
      nvtxRangeId_t mark =
          nvtxRangeStartA((std::string("TestOpCudaKernel: ") + unique_op_name_).c_str());
#endif
      cudaMemcpyAsync(output_tensor->data(), input_tensor->data(),
                      input_tensor->NumElements() * DataTypeSize(input_tensor->dtype()),
                      cudaMemcpyDeviceToDevice, stream);
#ifdef USE_NVTX
      nvtxRangeEnd(mark);
#endif
      done();
    };

    if (std::is_same<Device, CPUDevice>::value) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(std::move(work_func));
    } else if (std::is_same<Device, GPUDevice>::value) {
      auto stream = ctx->op_device_context()->stream();
      ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(stream,
                                                                          std::move(work_func));
    } else {
      ctx->SetStatus(errors::Aborted("Not supported Device Type"));
      done();
      return;
    }
  }

 private:
  std::mutex mu_;
  std::string unique_op_name_;
};
#else
template <typename Device>
class TestOp : public OpKernel {
 public:
  explicit TestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
    Tensor* y_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor->shape(), &y_tensor));

    std::cout << "\n[INFO]: input numelements = " << x_tensor->NumElements()
              << ", on thread: " << std::this_thread::get_id() << std::endl;

    int init_flag = 0;
    CK_MPI(ctx, MPI_Initialized(&init_flag));
    if (1 == init_flag) {
      std::this_thread::sleep_for(std::chrono::seconds(3));
      std::cout << "\n[INFO]: MPI has been Initialized." << std::endl;

      int rank = 0;
      CK_MPI(ctx, MPI_Comm_rank(MPI_COMM_WORLD, &rank));
      if (0 == rank) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
      }

      std::cout << "\n[INFO]: " << rank << " enter mpi barrier." << std::endl;
      CK_MPI(ctx, MPI_Barrier(MPI_COMM_WORLD));
      std::cout << "\n[INFO]: " << rank << " exit mpi barrier." << std::endl;

    } else {
      std::cout << "\n[INFO]: MPI has not been Initialized." << std::endl;
    }

    std::cout << "\n[INFO]: thread " << std::this_thread::get_id() << " done" << std::endl;
  }
};
#endif

REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_GPU), TestOp<GPUDevice>);
// REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_CPU),
//                         TestOp<CPUDevice>);

}  // namespace tensorflow