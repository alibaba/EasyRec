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

#include <cub/cub.cuh>

#include "common.cuh"
#include "common/include/conversion_kernels.cuh"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

template <typename KeyType, typename ValueType>
class CsrConversionDistributed : public Operation {
 public:
  explicit CsrConversionDistributed(ConstructionContext_t context)
      : Operation(context),
        resource_mgr_(context->get_resource_mgr()),
        slot_num_(context->get_slot_num()),
        max_nnz_(context->get_max_nnz()) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    binary_flags_.reserve(local_gpu_count);
    cub_d_temp_storage_.reserve(local_gpu_count);
    cub_coo_indices_output_.reserve(local_gpu_count);
    cub_values_output_.reserve(local_gpu_count);
    cub_host_num_selected_.reserve(local_gpu_count);
    cub_dev_num_selected_.reserve(local_gpu_count);
    cusparse_csr_row_offsets_output_.reserve(local_gpu_count);
    csr_row_offsets_cast_.reserve(local_gpu_count);
  }

  void allocate_forward_spaces() override {
    const size_t global_batch_size = base_context()->get_global_batch_size();
    for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
      auto &buffer = base_context()->get_buffer(dev_id);
      auto &host_buffer = base_context()->get_host_buffer(dev_id);
      {
        Tensor2<bool> binary_flag;
        buffer->reserve({1, global_batch_size * slot_num_ * max_nnz_}, &binary_flag);
        binary_flags_.push_back(binary_flag);
      }
      {
        Tensor2<int32_t> cub_coo_indices_output;
        buffer->reserve({1, global_batch_size * slot_num_ * max_nnz_}, &cub_coo_indices_output);
        cub_coo_indices_output_.push_back(cub_coo_indices_output);
      }
      {
        Tensor2<KeyType> cub_values_output;
        buffer->reserve({1, global_batch_size * slot_num_ * max_nnz_}, &cub_values_output);
        cub_values_output_.push_back(cub_values_output);
      }
      {
        Tensor2<size_t> cub_host_num_selected;
        host_buffer->reserve({1, 1}, &cub_host_num_selected);
        cub_host_num_selected_.push_back(cub_host_num_selected);
      }
      {
        Tensor2<size_t> cub_dev_num_selected;
        buffer->reserve({1, 1}, &cub_dev_num_selected);
        cub_dev_num_selected_.push_back(cub_dev_num_selected);
      }
      {
        Tensor2<int32_t> cusparse_csr_row_offset_output;
        buffer->reserve({1, global_batch_size * slot_num_ + 1}, &cusparse_csr_row_offset_output);
        cusparse_csr_row_offsets_output_.push_back(cusparse_csr_row_offset_output);
      }
      {
        Tensor2<int64_t> csr_row_offset_cast;
        buffer->reserve({1, global_batch_size * slot_num_ + 1}, &csr_row_offset_cast);
        csr_row_offsets_cast_.push_back(csr_row_offset_cast);
      }
      {
        size_t size_0 = 0;
        CK_CUDA(cub::DeviceSelect::Flagged(
            (void *)nullptr, size_0, (KeyType *)nullptr, (bool *)nullptr, (KeyType *)nullptr,
            (size_t *)nullptr, static_cast<int32_t>(global_batch_size * slot_num_ * max_nnz_)));
        size_t size_1 = 0;
        CK_CUDA(cub::DeviceSelect::Flagged(
            (void *)nullptr, size_1, (int64_t *)nullptr, (bool *)nullptr, (int32_t *)nullptr,
            (size_t *)nullptr, static_cast<int32_t>(global_batch_size * slot_num_ * max_nnz_)));

        size_t size = (size_0 > size_1) ? size_0 : size_1;
        Tensor2<void> cub_d_temp_storage;
        buffer->reserve({size}, &cub_d_temp_storage);
        cub_d_temp_storage_.push_back(cub_d_temp_storage);
      }
    }  // for dev_id
  }

  void allocate_backward_spaces() override {
    // it does nothing
  }

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
    const auto &stream = local_gpu->get_stream();

    const auto &total_values = replica_context->input("total_values");
    const auto &total_row_indices = replica_context->input("total_row_indices");
    const auto &host_total_num_elements = replica_context->input("host_total_num_elements");

    // reset internal buffers
    reset(local_replica_id);

    // generate binary vector
    gen_binary_vector(global_replica_id,
                      /*values=*/total_values,
                      /*total_valid_num=*/host_total_num_elements,
                      /*binary_flag=*/binary_flags_[local_replica_id]);

    // choose valuse based on binary vector
    size_t total_valid_num = host_total_num_elements->GetPtrWithType<size_t>()[0];
    size_t size = cub_d_temp_storage_[local_replica_id].get_size_in_bytes();
    CK_CUDA(cub::DeviceSelect::Flagged(
        /*d_temp_storage=*/cub_d_temp_storage_[local_replica_id].get_ptr(),
        /*temp_storage_bytes=*/size,
        /*d_in=*/total_values->GetPtrWithType<KeyType>(),
        /*d_flags=*/binary_flags_[local_replica_id].get_ptr(),
        /*d_out=*/cub_values_output_[local_replica_id].get_ptr(),
        /*d_num_selected_out=*/cub_dev_num_selected_[local_replica_id].get_ptr(),
        /*num_iterms=*/total_valid_num, stream));

    // copy num_selected (nnz) to host
    CK_CUDA(cudaMemcpyAsync(cub_host_num_selected_[local_replica_id].get_ptr(),
                            cub_dev_num_selected_[local_replica_id].get_ptr(),
                            cub_dev_num_selected_[local_replica_id].get_size_in_bytes(),
                            cudaMemcpyDeviceToHost, stream));
    CK_CUDA(cudaStreamSynchronize(stream));

    // choose row_indices based on binary vector
    CK_CUDA(cub::DeviceSelect::Flagged(
        /*d_temp_storage=*/cub_d_temp_storage_[local_replica_id].get_ptr(),
        /*temp_storage_bytes=*/size,
        /*d_in=*/total_row_indices->GetPtrWithType<int64_t>(),
        /*d_flags=*/binary_flags_[local_replica_id].get_ptr(),
        /*d_out=*/cub_coo_indices_output_[local_replica_id].get_ptr(),
        /*d_num_selected_out=*/cub_dev_num_selected_[local_replica_id].get_ptr(),
        /*num_iterms=*/total_valid_num, stream));

    // convert COO row_indices to CSR row_offsets.
    size_t rows_num = binary_flags_[local_replica_id].get_num_elements() / max_nnz_;
    CK_CUSPARSE(cusparseXcoo2csr(
        /*handle=*/local_gpu->get_cusparse(),
        /*cooRowInd=*/cub_coo_indices_output_[local_replica_id].get_ptr(),
        /*nnz=*/static_cast<int32_t>(cub_host_num_selected_[local_replica_id].get_ptr()[0]),
        /*m=*/rows_num,
        /*csrRowPtr=*/cusparse_csr_row_offsets_output_[local_replica_id].get_ptr(),
        CUSPARSE_INDEX_BASE_ZERO));

    // cast row_offset dtype
    auto op = [] __device__(int value) { return static_cast<int64_t>(value); };
    transform_array<<<local_gpu->get_sm_count() * 2, 1024, 0, stream>>>(
        cusparse_csr_row_offsets_output_[local_replica_id].get_ptr(),
        csr_row_offsets_cast_[local_replica_id].get_ptr(), rows_num + 1, op);

    // set outputs
    replica_context->set_output("replica_csr_values", cub_values_output_[local_replica_id]);
    replica_context->set_output("replica_row_offset", csr_row_offsets_cast_[local_replica_id]);

    auto& host_nnz = replica_context->output("replica_host_nnz");
    host_nnz->GetPtrWithType<size_t>()[0] = static_cast<size_t>(
            cub_host_num_selected_[local_replica_id].get_ptr()[0]);
  }

  void backward(const Context_t &replica_context) override {
    // it does nothing
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t slot_num_;
  const size_t max_nnz_;

  Tensors2<bool> binary_flags_;
  Tensors2<void> cub_d_temp_storage_;
  Tensors2<int32_t> cub_coo_indices_output_;
  Tensors2<KeyType> cub_values_output_; 
  Tensors2<size_t> cub_host_num_selected_;
  Tensors2<size_t> cub_dev_num_selected_;
  Tensors2<int32_t> cusparse_csr_row_offsets_output_;
  Tensors2<int64_t> csr_row_offsets_cast_;  // always int64, cause coo-indices always int64

  void reset(const size_t local_replica_id) {
    const auto &stream = resource_mgr_->get_local_gpu(local_replica_id)->get_stream();

    CK_CUDA(cudaMemsetAsync(binary_flags_[local_replica_id].get_ptr(), 0,
                            binary_flags_[local_replica_id].get_size_in_bytes(), stream));
    CK_CUDA(cudaMemsetAsync(cub_coo_indices_output_[local_replica_id].get_ptr(), 0,
                            cub_coo_indices_output_[local_replica_id].get_size_in_bytes(), stream));
    CK_CUDA(cudaMemsetAsync(cub_values_output_[local_replica_id].get_ptr(), 0,
                            cub_values_output_[local_replica_id].get_size_in_bytes(), stream));
    CK_CUDA(cudaMemsetAsync(cusparse_csr_row_offsets_output_[local_replica_id].get_ptr(), 0,
                            cusparse_csr_row_offsets_output_[local_replica_id].get_size_in_bytes(),
                            stream));
    CK_CUDA(cudaMemsetAsync(csr_row_offsets_cast_[local_replica_id].get_ptr(), 0,
                            csr_row_offsets_cast_[local_replica_id].get_size_in_bytes(), stream));
  }

 public:
  void gen_binary_vector(const size_t global_replica_id, const std::shared_ptr<Tensor> values,
                         const std::shared_ptr<Tensor> total_valid_num,
                         Tensor2<bool> &binary_flag) {
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    auto fn = [global_replica_id, global_gpu_count] __device__(KeyType value) -> bool {
      return (global_replica_id == value % global_gpu_count) ? true : false;
    };

    boolean_vector<<<local_gpu->get_sm_count() * 2, 1024, 0, local_gpu->get_stream()>>>(
        values->GetPtrWithType<KeyType>(), total_valid_num->GetPtrWithType<size_t>()[0], fn,
        binary_flag.get_ptr());
  }
};

REGISTER_OPERATION_BUILDER("csr_conversion_distributed", 
                           DataType::Int64,
                           DataType::Float32, 
                           CsrConversionDistributed<int64_t, float>);
REGISTER_OPERATION_BUILDER("csr_conversion_distributed", 
                           DataType::Int64,
                           DataType::Float16, 
                           CsrConversionDistributed<int64_t, __half>);
REGISTER_OPERATION_BUILDER("csr_conversion_distributed", 
                           DataType::Uint32,
                           DataType::Float32, 
                           CsrConversionDistributed<uint32_t, float>);
REGISTER_OPERATION_BUILDER("csr_conversion_distributed", 
                           DataType::Uint32,
                           DataType::Float16, 
                           CsrConversionDistributed<uint32_t, __half>);

}  // namespace SparseOperationKit