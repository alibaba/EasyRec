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

#include "common/include/forward_functions.h"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

template <typename ValueType>
class ReduceScatterDispatcher : public Dispatcher {
 public:
  explicit ReduceScatterDispatcher(ConstructionContext_t context)
      : Dispatcher(context),
        resource_mgr_(context->get_resource_mgr()),
        global_batch_size_(base_context()->get_global_batch_size()) {}

  void allocate_forward_spaces() override {}

  void allocate_backward_spaces() override {}

  void forward(const Context_t &replica_context, const bool training) override {
    const auto &embedding_features = replica_context->input("embedding_features");
    auto &replica_output = replica_context->output("replica_output");

    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);

    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    CK_NCCL(ncclReduceScatter(embedding_features->GetPtrWithType<ValueType>(),
                              replica_output->GetPtrWithType<ValueType>(),
                              replica_output->get_num_elements(), GetNCCLType<ValueType>(), 
                              ncclSum, local_gpu->get_nccl(), local_gpu->get_stream()));

    // do mean scale when Combiner::Mean is used.
    if (replica_context->has_internal_tensor("row_offset_allreduce_tensor")) {
      auto &row_offset_allreduce_tensor =
          replica_context->get_internal_tensor("row_offset_allreduce_tensor");
      const auto &row_offset_tensor = replica_context->input("replica_row_offset");

      CK_NCCL(ncclAllReduce(/*sendbuff=*/row_offset_tensor->GetPtrWithType<int64_t>(),
                            /*recvbuff=*/row_offset_allreduce_tensor->GetPtrWithType<int64_t>(),
                            /*count=*/row_offset_tensor->get_num_elements(),
                            /*datatype=*/ncclInt64,
                            /*op=*/ncclSum, local_gpu->get_nccl(), local_gpu->get_stream()));
      const size_t replica_batch_size = global_batch_size_ / resource_mgr_->get_global_gpu_count();
      const size_t rows_num_per_sample = base_context()->get_slot_num();
      const size_t emb_vec_size =
          replica_output->get_num_elements() / (replica_batch_size * rows_num_per_sample);
      const int64_t *row_offset = row_offset_allreduce_tensor->GetPtrWithType<int64_t>() +
                                  global_replica_id * replica_batch_size * rows_num_per_sample;
      do_forward_scale(/*batchsize_per_gpu=*/replica_batch_size,
                       /*slot_num=*/rows_num_per_sample,
                       /*embedding_vec_size=*/emb_vec_size,
                       /*row_offset=*/row_offset,
                       /*embedding_feature=*/replica_output->GetPtrWithType<ValueType>(),
                       local_gpu->get_stream());
    }  // if row_offset_allreduce_tensor_
  }

  void backward(const Context_t &replica_context) override {
    auto &replica_top_gradient = replica_context->input("replica_top_gradient");
    auto &embedding_feature = replica_context->input("embedding_features");

    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    CK_NCCL(ncclAllGather(/*sendbuff=*/replica_top_gradient->GetPtrWithType<ValueType>(),
                          /*recvbuff=*/embedding_feature->GetPtrWithType<ValueType>(),
                          /*sendcount=*/replica_top_gradient->get_num_elements(),
                          /*datatype=*/GetNCCLType<ValueType>(), 
                          local_gpu->get_nccl(), local_gpu->get_stream()));
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t global_batch_size_;
};

REGISTER_OUTPUT_DISPATHER_BUILDER("reduce_scatter_dispatcher", 
                                  DataType::Int64,
                                  DataType::Float32, 
                                  ReduceScatterDispatcher<float>);
REGISTER_OUTPUT_DISPATHER_BUILDER("reduce_scatter_dispatcher", 
                                  DataType::Int64,
                                  DataType::Float16, 
                                  ReduceScatterDispatcher<__half>);
REGISTER_OUTPUT_DISPATHER_BUILDER("reduce_scatter_dispatcher", 
                                  DataType::Uint32,
                                  DataType::Float32, 
                                  ReduceScatterDispatcher<float>);
REGISTER_OUTPUT_DISPATHER_BUILDER("reduce_scatter_dispatcher", 
                                  DataType::Uint32,
                                  DataType::Float16, 
                                  ReduceScatterDispatcher<__half>);

}  // namespace SparseOperationKit