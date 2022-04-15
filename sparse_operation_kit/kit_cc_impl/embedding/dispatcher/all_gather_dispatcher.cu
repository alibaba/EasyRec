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

#include "common.cuh"
#include "operation/operation_interface.h"
#include "common/include/forward_functions.h"

namespace SparseOperationKit {

template <typename T>
__global__ void move_data(const T *src_ptr, T *dst_ptr, size_t size, size_t *valid_nums,
                          size_t num_per_replica) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid = blockDim.x * gridDim.x;

  for (size_t i = gid; i < size; i += grid) {
    size_t which_replica = i / num_per_replica;
    size_t offset_in_replica = i % num_per_replica;
    if (offset_in_replica >= valid_nums[which_replica]) continue;
    size_t dst_offset = 0;
    for (size_t j = which_replica; j > 0; j--) {
      dst_offset += valid_nums[j - 1];
    }
    dst_ptr[dst_offset + offset_in_replica] = src_ptr[i];
  }
}

template <typename T>
__global__ void add_offset(T *ptr, size_t size, size_t *valid_nums, size_t num_per_replica,
                           size_t rows_num_per_replica) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid = blockDim.x * gridDim.x;
  for (size_t i = gid; i < size; i += grid) {
    size_t which_replica = i / num_per_replica;
    size_t offset_in_replica = i % num_per_replica;
    if (offset_in_replica >= valid_nums[which_replica]) continue;
    ptr[i] += (which_replica * rows_num_per_replica);
  }
}

template <typename KeyType, typename ValueType>
class AllGatherDispatcher : public Dispatcher {
 public:
  explicit AllGatherDispatcher(ConstructionContext_t context)
      : Dispatcher(context), resource_mgr_(context->get_resource_mgr()) {}

  void allocate_forward_spaces() override {
    auto rows_num_per_sample = base_context()->get_slot_num();
    auto max_nnz = base_context()->get_max_nnz();
    auto max_feature_num = base_context()->get_max_feature_num();

    const size_t global_batch_size = base_context()->get_global_batch_size();

    for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
      auto &buffer = base_context()->get_buffer(dev_id);
      auto &host_buffer = base_context()->get_host_buffer(dev_id);

      // reserve spaces for values buffers
      {
        Tensor2<KeyType> values_buffer;
        buffer->reserve({1, global_batch_size * rows_num_per_sample * max_nnz}, &values_buffer);
        values_buffers_.push_back(values_buffer);
        replica_num_elements_ = (global_batch_size * rows_num_per_sample * max_nnz) /
                                resource_mgr_->get_global_gpu_count();
      }
      // reserve spaces for indices buffers
      {
        // indices is always int64
        Tensor2<int64_t> indices_buffer;
        buffer->reserve({1, global_batch_size * rows_num_per_sample * max_nnz}, &indices_buffer);
        indices_buffers_.push_back(indices_buffer);
      }
      {
        Tensor2<int64_t> output_indices;
        buffer->reserve({1, global_batch_size * rows_num_per_sample * max_nnz}, &output_indices);
        output_indices_.push_back(output_indices);
      }
      {
        Tensor2<KeyType> output_values;
        buffer->reserve({1, global_batch_size * rows_num_per_sample * max_nnz}, &output_values);
        output_values_.push_back(output_values);
      }
      // reserve spaces for num elements
      {
        Tensor2<size_t> host_num_element;
        host_buffer->reserve({1, 1}, &host_num_element);
        host_num_elements_.push_back(host_num_element);

        Tensor2<size_t> num_element;
        buffer->reserve({1, resource_mgr_->get_global_gpu_count()}, &num_element);
        num_elements_.push_back(num_element);
      }
      {
        Tensor2<size_t> total_valid_num;
        host_buffer->reserve({1, 1}, &total_valid_num);
        total_valid_num_.push_back(total_valid_num);
      }
    }  // for dev_id
  }

  void allocate_backward_spaces() override {}

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    auto &replica_values = replica_context->input("replica_values");
    auto &replica_row_indices = replica_context->input("replica_indices");

    CK_CUDA(cudaMemcpyAsync(
        values_buffers_[local_replica_id].get_ptr() + (global_replica_id * replica_num_elements_),
        replica_values->GetPtrWithType<void>(), replica_values->get_size_in_bytes(),
        cudaMemcpyDeviceToDevice, local_gpu->get_stream()));
    CK_CUDA(cudaMemcpyAsync(
        indices_buffers_[local_replica_id].get_ptr() + (global_replica_id * replica_num_elements_),
        replica_row_indices->GetPtrWithType<void>(), replica_row_indices->get_size_in_bytes(),
        cudaMemcpyDeviceToDevice, local_gpu->get_stream()));

    host_num_elements_[local_replica_id].get_ptr()[0] = replica_values->get_num_elements();
    CK_CUDA(cudaMemcpyAsync(num_elements_[local_replica_id].get_ptr() + global_replica_id,
                            host_num_elements_[local_replica_id].get_ptr(), sizeof(size_t) * 1,
                            cudaMemcpyHostToDevice, local_gpu->get_stream()));

    CK_NCCL(ncclGroupStart());
    CK_NCCL(ncclAllGather(
        values_buffers_[local_replica_id].get_ptr() + (global_replica_id * replica_num_elements_),
        values_buffers_[local_replica_id].get_ptr(), replica_num_elements_, GetNCCLType<KeyType>(),
        local_gpu->get_nccl(), local_gpu->get_stream()));
    CK_NCCL(ncclAllGather(
        indices_buffers_[local_replica_id].get_ptr() + (global_replica_id * replica_num_elements_),
        indices_buffers_[local_replica_id].get_ptr(), replica_num_elements_, ncclInt64,
        local_gpu->get_nccl(), local_gpu->get_stream()));
    CK_NCCL(ncclAllGather(num_elements_[local_replica_id].get_ptr() + global_replica_id,
                          num_elements_[local_replica_id].get_ptr(), 1, ncclUint64,
                          local_gpu->get_nccl(), local_gpu->get_stream()));
    CK_NCCL(ncclGroupEnd());

    // make the memory successive
    move_data<<<local_gpu->get_sm_count(), 1024, 0, local_gpu->get_stream()>>>(
        values_buffers_[local_replica_id].get_ptr(), output_values_[local_replica_id].get_ptr(),
        values_buffers_[local_replica_id].get_num_elements(),
        num_elements_[local_replica_id].get_ptr(), replica_num_elements_);

    // calculate the offset of row_indices
    add_offset<<<local_gpu->get_sm_count(), 1024, 0, local_gpu->get_stream()>>>(
        indices_buffers_[local_replica_id].get_ptr(),
        indices_buffers_[local_replica_id].get_num_elements(),
        num_elements_[local_replica_id].get_ptr(), replica_num_elements_,
        replica_num_elements_ / base_context()->get_max_nnz());

    // make the memory successive
    move_data<<<local_gpu->get_sm_count(), 1024, 0, local_gpu->get_stream()>>>(
        indices_buffers_[local_replica_id].get_ptr(), output_indices_[local_replica_id].get_ptr(),
        indices_buffers_[local_replica_id].get_num_elements(),
        num_elements_[local_replica_id].get_ptr(), replica_num_elements_);
    reduce_sum<<<1, 1, 0, local_gpu->get_stream()>>>(
        num_elements_[local_replica_id].get_ptr(),
        num_elements_[local_replica_id].get_num_elements(),
        total_valid_num_[local_replica_id].get_ptr());
    // copy back to host
    resource_mgr_->sync_gpu(local_replica_id);
    CK_CUDA(cudaMemcpyAsync(host_num_elements_[local_replica_id].get_ptr(),
                            total_valid_num_[local_replica_id].get_ptr(),
                            total_valid_num_[local_replica_id].get_size_in_bytes(),
                            cudaMemcpyDeviceToHost, local_gpu->get_stream()));
    resource_mgr_->sync_gpu(local_replica_id);

    // set output for this operation
    replica_context->set_output("total_values", output_values_[local_replica_id]);
    replica_context->set_output("total_row_indices", output_indices_[local_replica_id]);
    replica_context->set_output("dev_total_num_elements", num_elements_[local_replica_id]);
    replica_context->set_output("host_total_num_elements", host_num_elements_[local_replica_id]);
  }

  void backward(const Context_t &replica_context) override {
    // it does nothing
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  size_t replica_num_elements_ = 0;

  Tensors2<KeyType> values_buffers_;
  Tensors2<KeyType> output_values_;
  Tensors2<int64_t> indices_buffers_; // always int64
  Tensors2<int64_t> output_indices_; // always int64
  Tensors2<size_t> host_num_elements_;
  Tensors2<size_t> num_elements_;
  Tensors2<size_t> total_valid_num_;
};

REGISTER_INPUT_DISPATCHER_BUILDER("all_gather_dispatcher", 
                                  DataType::Int64,
                                  DataType::Float32, 
                                  AllGatherDispatcher<int64_t, float>);
REGISTER_INPUT_DISPATCHER_BUILDER("all_gather_dispatcher", 
                                  DataType::Int64,
                                  DataType::Float16, 
                                  AllGatherDispatcher<int64_t, __half>);
REGISTER_INPUT_DISPATCHER_BUILDER("all_gather_dispatcher", 
                                  DataType::Uint32,
                                  DataType::Float32, 
                                  AllGatherDispatcher<uint32_t, float>);
REGISTER_INPUT_DISPATCHER_BUILDER("all_gather_dispatcher", 
                                  DataType::Uint32,
                                  DataType::Float16, 
                                  AllGatherDispatcher<uint32_t, __half>);

}  // namespace SparseOperationKit