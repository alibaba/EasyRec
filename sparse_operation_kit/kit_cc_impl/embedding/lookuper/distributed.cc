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

#include "common/include/backward_functions.h"
#include "common/include/dumping_functions.h"
#include "common/include/forward_functions.h"
#include "hashtable/simple_hashtable.h"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

template <typename KeyType, typename ValueType>
class DistribtuedLookuper : public EmbeddingLookuper {
 public:
  explicit DistribtuedLookuper(ConstructionContext_t context, std::shared_ptr<ParamInterface> param)
      : EmbeddingLookuper(context, param),
        resource_mgr_(context->get_resource_mgr()),
        max_feature_num_(context->get_max_feature_num()),
        slot_num_(context->get_slot_num()),
        combiner_(context->get_combiner()),
        global_batch_size_(base_context()->get_global_batch_size()) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    hash_value_index_tensors_.reserve(local_gpu_count);
    embedding_feature_tensors_.reserve(local_gpu_count);
    wgrad_tensors_.reserve(local_gpu_count);
    if (combiner_ == CombinerType::Mean) row_offset_allreduce_tensors_.reserve(local_gpu_count);

    if (param->get_hashtable(0)->identical_mapping()) {
      // identical_mapping waste memory spaces, so that lookuper
      // will set its wanted hashtable for param
      const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
      auto stream = resource_mgr_->get_local_gpu(0)->get_stream();
      const size_t capacity = param->get_hashtable(0)->get_capacity(stream);
      HashFunctor_t hash_func = HashFunctors::Divisive<KeyType, size_t>::create(
          /*interval=*/global_gpu_count, /*capacity=*/capacity,
          /*global_replica_id=*/resource_mgr_->cal_global_id_from_local_id(0));
      auto hashtable = SimpleHashtable<KeyType, size_t>::create(capacity, hash_func);
      param->set_hashtable(hashtable);
    }  // if identical_mapping
  }

  void allocate_forward_spaces() override {
    size_t max_vocabulary_size_per_gpu = param_->get_max_vocabulary_size_per_gpu();
    size_t max_vocabulary_size_in_total =
        max_vocabulary_size_per_gpu * resource_mgr_->get_global_gpu_count();

    for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
      auto &buffer = base_context()->get_buffer(dev_id);
      // new hash table value (index) that get() from hashtable
      {
        Tensor2<size_t> tensor;
        buffer->reserve({1, global_batch_size_ * max_feature_num_}, &tensor);
        hash_value_index_tensors_.push_back(tensor);
      }
      // new embedding features reduced by hash table values.
      {
        Tensor2<ValueType> tensor;
        buffer->reserve({global_batch_size_ * slot_num_, param_->get_embedding_vec_size()},
                        &tensor);
        embedding_feature_tensors_.push_back(tensor);
      }
    }  // for dev_id
  }

  void allocate_backward_spaces() override {
    for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
      auto &buffer = base_context()->get_buffer(dev_id);
      // new wgrad used by backward
      {
        Tensor2<ValueType> tensor;
        buffer->reserve({global_batch_size_ * slot_num_, param_->get_embedding_vec_size()},
                        &tensor);
        wgrad_tensors_.push_back(tensor);
      }
      {
        if (CombinerType::Mean == combiner_) {
          Tensor2<int64_t> tensor;
          buffer->reserve({1, global_batch_size_ * slot_num_ + 1}, &tensor);
          row_offset_allreduce_tensors_.push_back(tensor);
        }  // if combiner_ == mean
      }

    }  // for dev_id
  }

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    const auto &replica_csr_values = replica_context->input("replica_csr_values");
    const auto &replica_row_offset = replica_context->input("replica_row_offset");
    const auto &replica_host_nnz = replica_context->input("replica_host_nnz");

    // get hash_value_index from hash_table by hash_key
    auto &hashtable = param_->get_hashtable(local_replica_id);
    if (training) {
      hashtable->get_insert(replica_csr_values->GetPtrWithType<KeyType>(),
                            hash_value_index_tensors_[local_replica_id].get_ptr(),
                            replica_host_nnz->GetPtrWithType<size_t>()[0], local_gpu->get_stream());
    } else {
      hashtable->get(replica_csr_values->GetPtrWithType<KeyType>(),
                     hash_value_index_tensors_[local_replica_id].get_ptr(),
                     replica_host_nnz->GetPtrWithType<size_t>()[0], local_gpu->get_stream());
    }

    replica_context->record_internal_tensor("replica_hash_value_index",
                                            hash_value_index_tensors_[local_replica_id],
                                            /*overwrite=*/true);

    // embedding vector looking up and do reduction
    switch (combiner_) {
      case CombinerType::Sum: {
        forward_sum(/*batch_size=*/global_batch_size_, slot_num_, param_->get_embedding_vec_size(),
                    /*row_offsets=*/replica_row_offset->GetPtrWithType<int64_t>(),
                    hash_value_index_tensors_[local_replica_id].get_ptr(),
                    /*hash_table_value=*/
                    param_->get_embedding_table_tensor(local_replica_id)->GetPtrWithType<float>(),
                    /*embedding_feature=*/embedding_feature_tensors_[local_replica_id].get_ptr(),
                    local_gpu->get_stream());
        break;
      }
      case CombinerType::Mean: {
        // delay mean scale after reduction-sum
        forward_sum(/*batch_size=*/global_batch_size_, slot_num_, param_->get_embedding_vec_size(),
                    /*row_offsets=*/replica_row_offset->GetPtrWithType<int64_t>(),
                    hash_value_index_tensors_[local_replica_id].get_ptr(),
                    /*hash_table_value=*/
                    param_->get_embedding_table_tensor(local_replica_id)->GetPtrWithType<float>(),
                    /*embedding_feature=*/embedding_feature_tensors_[local_replica_id].get_ptr(),
                    local_gpu->get_stream());
        replica_context->record_internal_tensor("row_offset_allreduce_tensor",
                                                row_offset_allreduce_tensors_[local_replica_id]);
        break;
      }
      default:
        throw std::runtime_error(ErrorBase + "Not supported combiner.");
    }  // switch combiner_

    // set outputs
    replica_context->set_output("embedding_features", embedding_feature_tensors_[local_replica_id]);
  }

  void backward(const Context_t &replica_context) override {
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &stream = resource_mgr_->get_local_gpu(local_replica_id)->get_stream();

    const auto &embedding_features = replica_context->input("embedding_features");

    switch (combiner_) {
      case CombinerType::Sum: {
        backward_sum(/*batch_size=*/global_batch_size_, slot_num_, param_->get_embedding_vec_size(),
                     /*top_grad=*/embedding_features->GetPtrWithType<ValueType>(),
                     /*wgrad=*/wgrad_tensors_[local_replica_id].get_ptr(), stream);
        break;
      }
      case CombinerType::Mean: {
        backward_mean(/*batch_size=*/global_batch_size_, slot_num_,
                      param_->get_embedding_vec_size(),
                      /*row_offset=*/row_offset_allreduce_tensors_[local_replica_id].get_ptr(),
                      /*top_grad=*/embedding_features->GetPtrWithType<ValueType>(),
                      /*wgrad=*/wgrad_tensors_[local_replica_id].get_ptr(), stream);
        break;
      }
      default:
        throw std::runtime_error(ErrorBase + "Not supported combiner.");
    }  // switch combiner_

    // set input grads
    const auto &replica_row_offset = replica_context->input("replica_row_offset");
    auto &replica_input_grad = replica_context->output("replica_input_grad");
    expand_input_grad(global_batch_size_, slot_num_, param_->get_embedding_vec_size(),
                      replica_row_offset->GetPtrWithType<int64_t>(),
                      wgrad_tensors_[local_replica_id].get_ptr(),
                      replica_input_grad->GetPtrWithType<ValueType>(), stream);

    // set hash_value index
    const auto &replica_hash_value_index =
        replica_context->get_internal_tensor("replica_hash_value_index");
    auto &value_index_tensor = replica_context->output("value_index_tensor");
    CK_CUDA(cudaMemcpyAsync(value_index_tensor->GetPtrWithType<int64_t>(),
                            replica_hash_value_index->GetPtrWithType<int64_t>(),
                            value_index_tensor->get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                            stream));
  }

  void save_params(std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
                   size_t &num_total_keys) const override {
    // this lookuper distribute keys to each GPU based on key % GPU_NUM
    save_params_helper<KeyType>(param_, resource_mgr_, keys, embedding_values, num_total_keys);
  }

  void restore_params(const std::shared_ptr<Tensor> &keys,
                      const std::shared_ptr<Tensor> &embedding_values,
                      const size_t num_total_keys) override {
    // this lookuper distribute keys to each GPU based on key % GPU_NUM
    restore_params_helper<KeyType>(param_, resource_mgr_, keys, embedding_values, num_total_keys);
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t max_feature_num_;
  const size_t slot_num_;
  CombinerType combiner_;
  const size_t global_batch_size_;

  // forward spaces
  Tensors2<size_t> hash_value_index_tensors_;
  Tensors2<ValueType> embedding_feature_tensors_;

  // backward spaces
  Tensors2<ValueType> wgrad_tensors_;
  Tensors2<int64_t> row_offset_allreduce_tensors_;
};

REGISTER_EMB_LOOKUPER_BUILDER("distributed", 
                              DataType::Int64,
                              DataType::Float32, 
                              DistribtuedLookuper<int64_t, float>);
REGISTER_EMB_LOOKUPER_BUILDER("distributed", 
                              DataType::Int64,
                              DataType::Float16, 
                              DistribtuedLookuper<int64_t, __half>);
REGISTER_EMB_LOOKUPER_BUILDER("distributed", 
                              DataType::Uint32,
                              DataType::Float32, 
                              DistribtuedLookuper<uint32_t, float>);
REGISTER_EMB_LOOKUPER_BUILDER("distributed", 
                              DataType::Uint32,
                              DataType::Float16, 
                              DistribtuedLookuper<uint32_t, __half>);

}  // namespace SparseOperationKit