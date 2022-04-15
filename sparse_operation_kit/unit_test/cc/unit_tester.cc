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

#include "operation/builder_container.h"
#include "operation/construction_context.h"
#include "operation/op_context.h"
#include "operation/operation.h"
#include "tensor_buffer/tf_tensor_wrapper.h"

namespace SparseOperationKit {

UnitTester::UnitTester(const std::shared_ptr<ResourcesManager>& resource_mgr)
    : resource_mgr_(resource_mgr) {
  buffers_.clear();
  buffers_.resize(resource_mgr_->get_local_gpu_count(), nullptr);
  host_buffers_.clear();
  host_buffers_.resize(resource_mgr_->get_local_gpu_count(), nullptr);

  for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
    buffers_[dev_id] = HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>::create();
    host_buffers_[dev_id] = HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>::create();
  }
}

UnitTester* UnitTester::instance(const std::shared_ptr<ResourcesManager>& resource_mgr) {
  static UnitTester ins(resource_mgr);
  return &ins;
}

void UnitTester::operator delete(void*) {
  throw std::domain_error("This pointer cannot be manually deleted.");
}

void UnitTester::test_all_gather_dispatcher(
    const size_t rows_num_per_sample, const size_t max_nnz, size_t const global_batch_size,
    const size_t global_replica_id, const tensorflow::Tensor* values_tensor,
    const tensorflow::Tensor* indices_tensor, tensorflow::Tensor* values_out_tensor,
    tensorflow::Tensor* indices_out_tensor, tensorflow::Tensor* num_elements_tensor,
    tensorflow::Tensor* total_valid_num_tensor) {
#ifdef SOK_ASYNC
  resource_mgr_->event_record(global_replica_id, EventRecordType::RDLFramework,
                              /*event_name=*/"AllGatherUnitTest_begin");
#endif
  auto key_dtype = DataType::Int64;

  SparseConstructionContext_t base_context = SparseConstructionContext::create(
      resource_mgr_, buffers_, host_buffers_,
      /*replica_batch_size=*/global_batch_size / resource_mgr_->get_global_gpu_count(),
      rows_num_per_sample, max_nnz, /*max_feature_num=*/rows_num_per_sample * max_nnz,
      /*combiner=*/CombinerType::Mean, /*key_dtype=*/key_dtype,
      /*compute_dtype=*/DataType::Float32, /*param=*/nullptr);

  auto builder =
      InputContainer::instance("input_dispatcher_builders")
            ->get_builder({"all_gather_dispatcher", key_dtype, DataType::Float32});
  static std::shared_ptr<Dispatcher> dispatcher = builder->produce(base_context);

  auto init = [this, rows_num_per_sample, max_nnz]() {
    dispatcher->allocate_forward_spaces();
    dispatcher->allocate_backward_spaces();
  };
  resource_mgr_->blocking_call_once(init);

  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);

  try_allocate_memory(local_replica_id);

  Context_t replica_context = Context::create(global_replica_id);

  replica_context->set_input(
      "replica_values", TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(values_tensor)),
      /*overwrite=*/true);
  replica_context->set_input(
      "replica_indices", TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(indices_tensor)),
      /*overwrite=*/true);

  dispatcher->forward(replica_context, /*training=*/true);

  const auto total_values = replica_context->output("total_values");
  const auto total_row_indices = replica_context->output("total_row_indices");
  const auto dev_total_num_elements = replica_context->output("dev_total_num_elements");
  const auto host_total_num_elements = replica_context->output("host_total_num_elements");

  auto local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
  CK_CUDA(cudaMemcpyAsync(values_out_tensor->data(), total_values->GetPtrWithType<void>(),
                          total_values->get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                          local_gpu->get_stream()));
  CK_CUDA(cudaMemcpyAsync(indices_out_tensor->data(), total_row_indices->GetPtrWithType<void>(),
                          total_row_indices->get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                          local_gpu->get_stream()));
  CK_CUDA(cudaMemcpyAsync(num_elements_tensor->data(),
                          dev_total_num_elements->GetPtrWithType<void>(),
                          dev_total_num_elements->get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                          local_gpu->get_stream()));
  CK_CUDA(cudaMemcpyAsync(
      total_valid_num_tensor->data(), host_total_num_elements->GetPtrWithType<void>(),
      host_total_num_elements->get_size_in_bytes(), cudaMemcpyDefault, local_gpu->get_stream()));
#ifdef SOK_ASYNC
  resource_mgr_->event_record(global_replica_id, EventRecordType::RMyself,
                              /*event_name=*/"AllGatherUnitTest_end");
#endif
}

void UnitTester::test_csr_conversion_distributed(
    const size_t global_replica_id, const size_t global_batch_size, const size_t slot_num,
    const size_t max_nnz, const tensorflow::Tensor* values_tensor,
    const tensorflow::Tensor* row_indices_tensor, const tensorflow::Tensor* num_elements_tensor,
    const tensorflow::Tensor* total_valid_num_tensor, tensorflow::Tensor* replcia_values_tensor,
    tensorflow::Tensor* replica_csr_row_offsets_tensor, tensorflow::Tensor* replica_nnz_tensor) {
#ifdef SOK_ASYNC
  resource_mgr_->event_record(global_replica_id, EventRecordType::RDLFramework,
                              /*event_name=*/"CSRConversionDistributedUnitTest_begin");
#endif
  SparseConstructionContext_t base_context = SparseConstructionContext::create(
      resource_mgr_, buffers_, host_buffers_,
      /*replica_batch_size=*/global_batch_size / resource_mgr_->get_global_gpu_count(), slot_num,
      max_nnz, /*max_feature_num=*/slot_num * max_nnz,
      /*combiner=*/CombinerType::Mean, /*key_dtype=*/DataType::Int64,
      /*compute_dtype=*/DataType::Float32, /*param=*/nullptr);

  auto builder =
      OperationContainer::instance("operation_builders")
          ->get_builder({"csr_conversion_distributed", DataType::Int64, DataType::Float32});
  static std::shared_ptr<Operation> csr_conver = builder->produce(base_context);

  auto init = [this, slot_num, max_nnz]() {
    csr_conver->allocate_forward_spaces();
    csr_conver->allocate_backward_spaces();
  };
  resource_mgr_->blocking_call_once(init);

  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);

  try_allocate_memory(local_replica_id);

  Context_t replica_context = Context::create(global_replica_id);

  replica_context->set_input(
      "total_values", TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(values_tensor)),
      /*overwrite=*/true);
  replica_context->set_input(
      "total_row_indices",
      TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(row_indices_tensor)),
      /*overwrite=*/true);
  replica_context->set_input(
      "dev_total_num_elements",
      TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(num_elements_tensor)),
      /*overwrite=*/true);
  replica_context->set_input(
      "host_total_num_elements",
      TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(total_valid_num_tensor)),
      /*overwrite=*/true);
  replica_context->set_output(
      "replica_host_nnz",
      TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(replica_nnz_tensor)),
      /*overwrite=*/true);

  csr_conver->forward(replica_context, /*training=*/true);

  const auto replica_csr_values = replica_context->output("replica_csr_values");
  const auto replica_row_offset = replica_context->output("replica_row_offset");

  auto local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
  CK_CUDA(cudaMemcpyAsync(replcia_values_tensor->data(), replica_csr_values->GetPtrWithType<void>(),
                          replica_csr_values->get_size_in_bytes(), cudaMemcpyDefault,
                          local_gpu->get_stream()));
  CK_CUDA(cudaMemcpyAsync(
      replica_csr_row_offsets_tensor->data(), replica_row_offset->GetPtrWithType<void>(),
      replica_row_offset->get_size_in_bytes(), cudaMemcpyDefault, local_gpu->get_stream()));
#ifdef SOK_ASYNC
  resource_mgr_->event_record(global_replica_id, EventRecordType::RMyself,
                              /*event_name=*/"CSRConversionDistributedUnitTest_end");
#endif
}

void UnitTester::test_reduce_scatter_dispatcher(const size_t global_replica_id,
                                                const size_t global_batch_size,
                                                const size_t slot_num, const size_t max_nnz,
                                                const tensorflow::Tensor* input,
                                                tensorflow::Tensor* output) {
#ifdef SOK_ASYNC
  resource_mgr_->event_record(global_replica_id, EventRecordType::RDLFramework,
                              /*event_name=*/"ReduceScatterUnitTest_begin");
#endif
  SparseConstructionContext_t base_context = SparseConstructionContext::create(
      resource_mgr_, buffers_, host_buffers_,
      /*replica_batch_size=*/global_batch_size / resource_mgr_->get_global_gpu_count(), slot_num,
      max_nnz, /*max_feature_num=*/slot_num * max_nnz,
      /*combiner=*/CombinerType::Mean, /*key_dtype=*/DataType::Int64, 
      /*compute_dtype=*/DataType::Float32, /*param=*/nullptr);

  auto builder = OutputContainer::instance("output_dispatcher_builders")
                     ->get_builder({"reduce_scatter_dispatcher", DataType::Int64, DataType::Float32});
  static std::shared_ptr<Dispatcher> reduce_scatter_dispatcher = builder->produce(base_context);

  auto init = [this, slot_num, max_nnz]() {
    reduce_scatter_dispatcher->allocate_forward_spaces();
    reduce_scatter_dispatcher->allocate_backward_spaces();
  };
  resource_mgr_->blocking_call_once(init);

  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);

  try_allocate_memory(local_replica_id);

  Context_t replica_context = Context::create(global_replica_id);

  replica_context->set_input("embedding_features",
                             TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(input)),
                             /*overwrite=*/true);
  replica_context->set_output("replica_output", TFTensorWrapper::create(output),
                              /*overwrite=*/true);

  reduce_scatter_dispatcher->forward(replica_context, /*training=*/true);
#ifdef SOK_ASYNC
  resource_mgr_->event_record(global_replica_id, EventRecordType::RMyself,
                              /*event_name=*/"ReduceScatterUnitTest_end");
#endif
}

void UnitTester::try_allocate_memory(const size_t local_replica_id) const {
  static bool allocated = false;
  if (!allocated) {
    buffers_[local_replica_id]->allocate();
    host_buffers_[local_replica_id]->allocate();
    auto helper = []() {
      allocated = true;
      MESSAGE("Allocated unit tester buffer.");
    };
    resource_mgr_->blocking_call_once(helper);
  }
}

}  // namespace SparseOperationKit