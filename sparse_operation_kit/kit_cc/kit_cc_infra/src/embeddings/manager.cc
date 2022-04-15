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

#include "embeddings/manager.h"

#include <thread>

#include "operation/builder_container.h"
#include "operation/construction_context.h"
#include "optimizer/grad_update_preparer.h"

namespace SparseOperationKit {

EmbeddingManager::EmbeddingManager(const std::shared_ptr<ResourcesManager>& resource_mgr)
    : resource_mgr_(resource_mgr), resized_(false), mu_() {}

std::shared_ptr<EmbeddingManager> EmbeddingManager::Create(
    const std::shared_ptr<ResourcesManager>& resource_mgr) {
  return std::shared_ptr<EmbeddingManager>(new EmbeddingManager(resource_mgr));
}

void EmbeddingManager::init(const size_t global_replica_id, const size_t global_batch_size) {
  // resize buffers_ vector
  auto helper = [this, &global_batch_size]() {
    if (resized_) throw std::runtime_error(ErrorBase + "EmbeddingManager had been initialized.");
    buffers_.resize(resource_mgr_->get_local_gpu_count());
    host_buffers_.resize(resource_mgr_->get_local_gpu_count());

    if (global_batch_size < resource_mgr_->get_global_gpu_count())
      throw std::runtime_error(ErrorBase +
                               "Too small global_batch_size, "
                               "which is less than global_gpu_count");
    if (global_batch_size % resource_mgr_->get_global_gpu_count() != 0)
      throw std::runtime_error(ErrorBase +
                               "global_batch_size is not divisible by global_gpu_count.");

    global_batch_size_ = global_batch_size;

    resized_ = true;
  };
  // new GeneralBuffer object for local GPU by its own.
  resource_mgr_->blocking_call_once(helper);

  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
  buffers_[local_replica_id] = HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>::create();
  host_buffers_[local_replica_id] = HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>::create();
}

void EmbeddingManager::create_embedding(
    const std::shared_ptr<ParamInterface>& param, const std::string input_dispatcher,
    const std::vector<std::string> input_dispatcher_subsequent_ops,
    const std::string embedding_executor, const std::string output_dispatcher,
    const std::vector<std::string> output_dispatcher_subsequent_ops, const size_t slot_num,
    const size_t max_nnz, const size_t max_feature_num, const CombinerType combiner,
    const DataType compute_dtype,
    std::shared_ptr<EmbeddingLayer>& embedding) {
  // get key_dtype from parameter
  const DataType key_dtype = param->key_dtype();

  // create sparse construction context
  SparseConstructionContext_t construction_context = SparseConstructionContext::create(
      resource_mgr_, buffers_, host_buffers_, get_replica_batch_size(), slot_num, max_nnz,
      max_feature_num, combiner, key_dtype, compute_dtype, param);

  // produce input_dispatcher
  auto input_dis_builder =
      InputContainer::instance("input_dispatcher_builders")->get_builder(
                              {input_dispatcher, key_dtype, compute_dtype});
  std::shared_ptr<Dispatcher> _input_dispatcher = input_dis_builder->produce(construction_context);
  _input_dispatcher->set_op_name(input_dispatcher);

  // produce input_dispatcher subsequent operations
  for (auto op_name : input_dispatcher_subsequent_ops) {
    auto builder = OperationContainer::instance("operation_builders")->get_builder(
                                                {op_name, key_dtype, compute_dtype});
    std::shared_ptr<Operation> _operation = builder->produce(construction_context);
    _operation->set_op_name(op_name);
    _input_dispatcher->set_next(_operation);  // link this op to input_dispatcher
  }                                           // for op_name in input_dispatcher_subsequent_ops

  // produce embedding executor
  auto emb_lookuper_builder =
      LookuperContainer::instance("embedding_lookuper_builders")->get_builder(
                                {embedding_executor, key_dtype, compute_dtype});
  std::shared_ptr<EmbeddingLookuper> _embedding_lookuper =
      emb_lookuper_builder->produce(construction_context, param);
  _embedding_lookuper->set_op_name(embedding_executor);

  // produce output_dispatcher
  auto output_dis_builder =
      OutputContainer::instance("output_dispatcher_builders")->get_builder(
                              {output_dispatcher, key_dtype, compute_dtype});
  std::shared_ptr<Dispatcher> _output_dispatcher =
      output_dis_builder->produce(construction_context);
  _output_dispatcher->set_op_name(output_dispatcher);

  // produce output_dispatcher subsequent operations
  for (auto op_name : output_dispatcher_subsequent_ops) {
    auto builder = OperationContainer::instance("operation_builders")->get_builder(
                                                {op_name, key_dtype, compute_dtype});
    std::shared_ptr<Operation> _operation = builder->produce(construction_context);
    _operation->set_op_name(op_name);
    _output_dispatcher->set_next(_operation);  // link this op to output_dispatcher
  }                                            // for op_name in output_dispatcher_subsequent_ops

  // link these components to an embedding layer
  std::shared_ptr<EmbeddingLayer> embedding_temp = EmbeddingLayer::create(
      _input_dispatcher, _embedding_lookuper, _output_dispatcher, construction_context);
  {
    std::lock_guard<std::mutex> lock(mu_);
    embedding_temp->allocate_forward_spaces();
    if (param->trainable()) embedding_temp->allocate_backward_spaces();
    param->set_user(embedding_temp);
    embeddings_.emplace_back(embedding_temp);
    embedding = embedding_temp;
    // create compute context for this embedding layer
    create_contexts(embedding);
  }
}

void EmbeddingManager::create_embedding(
    const std::shared_ptr<ParamInterface>& param, const std::string input_dispatcher,
    const std::vector<std::string> input_dispatcher_subsequent_ops,
    const std::string embedding_lookuper, const std::string output_dispatcher,
    const std::vector<std::string> output_dispatcher_subsequent_ops, const size_t slot_num,
    const size_t nnz_per_slot, const DataType compute_dtype,
    std::shared_ptr<EmbeddingLayer>& embedding) {
  // get key_dtype from parameter
  const DataType key_dtype = param->key_dtype();

  // create dense construction_context
  DenseConstructionContext_t construction_context =
      DenseConstructionContext::create(resource_mgr_, buffers_, host_buffers_,
                                       get_replica_batch_size(), slot_num, nnz_per_slot, 
                                       key_dtype, compute_dtype, param);

  // produce input_dispatcher
  auto input_dis_builder =
      InputContainer::instance("input_dispatcher_builders")->get_builder(
                              {input_dispatcher, key_dtype, compute_dtype});
  std::shared_ptr<Dispatcher> _input_dispatcher = input_dis_builder->produce(construction_context);
  _input_dispatcher->set_op_name(input_dispatcher);

  // produce input_dispatcher subsequent operations
  for (auto op_name : input_dispatcher_subsequent_ops) {
    auto builder = OperationContainer::instance("operation_builders")->get_builder(
                                                {op_name, key_dtype, compute_dtype});
    std::shared_ptr<Operation> _operation = builder->produce(construction_context);
    _operation->set_op_name(op_name);
    _input_dispatcher->set_next(_operation);  // link this op to input_dispatcher
  }                                           // for op_name in input_dispatcher_subsequent_ops

  // produce embedding lookuper
  auto emb_lookuper_builder =
      LookuperContainer::instance("embedding_lookuper_builders")->get_builder(
                                  {embedding_lookuper, key_dtype, compute_dtype});
  std::shared_ptr<EmbeddingLookuper> _embedding_lookuper =
      emb_lookuper_builder->produce(construction_context, param);
  _embedding_lookuper->set_op_name(embedding_lookuper);

  // produce output_dispatcher
  auto output_dis_builder =
      OutputContainer::instance("output_dispatcher_builders")->get_builder(
                              {output_dispatcher, key_dtype, compute_dtype});
  std::shared_ptr<Dispatcher> _output_dispatcher =
      output_dis_builder->produce(construction_context);
  _output_dispatcher->set_op_name(output_dispatcher);

  // produce output_dispatcher subsequent operations
  for (auto op_name : output_dispatcher_subsequent_ops) {
    auto builder = OperationContainer::instance("operation_builders")->get_builder(
                                                {op_name, key_dtype, compute_dtype});
    std::shared_ptr<Operation> _operation = builder->produce(construction_context);
    _operation->set_op_name(op_name);
    _output_dispatcher->set_next(_operation);  // link this op to output_dispatcher
  }

  // link these components to an embedding layer.
  std::shared_ptr<EmbeddingLayer> embedding_temp = DenseEmbeddingLayer::create(
      _input_dispatcher, _embedding_lookuper, _output_dispatcher, construction_context);
  {
    std::lock_guard<std::mutex> lock(mu_);
    embedding_temp->allocate_forward_spaces();
    if (param->trainable()) embedding_temp->allocate_backward_spaces();
    param->set_user(embedding_temp);
    embeddings_.emplace_back(embedding_temp);
    embedding = embedding_temp;
    // create context for this embedding layer
    create_contexts(embedding);
  }
}

void EmbeddingManager::create_contexts(std::shared_ptr<EmbeddingLayer> embedding) {
  auto iter = embedding_contexts_.find(embedding);
  if (iter != embedding_contexts_.end())
    throw std::runtime_error(ErrorBase +
                             "There already exists contexts for this embedding. "
                             "This might be caused by you called create_contexts() "
                             "for the same embedding layer more than once.");

  std::vector<Context_t> contexts;
  contexts.resize(resource_mgr_->get_local_gpu_count());
  for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); dev_id++) {
    const size_t global_replica_id = resource_mgr_->cal_global_id_from_local_id(dev_id);
    contexts[dev_id] = Context::create(global_replica_id);
  }
  embedding_contexts_.emplace(std::make_pair(embedding, contexts));
}

Context_t& EmbeddingManager::get_context(const std::shared_ptr<EmbeddingLayer>& embedding,
                                         const size_t global_replica_id) {
  auto iter = embedding_contexts_.find(embedding);
  if (iter == embedding_contexts_.end())
    throw std::runtime_error(ErrorBase + "Cannot find context for this embedding.");
  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
  return iter->second[local_replica_id];
}

void EmbeddingManager::allocate_memory(const size_t global_replica_id) {
  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
  if (local_replica_id >= buffers_.size())
    throw std::runtime_error(ErrorBase +
                             "The local_replica_id is out of the range of buffers.size().");

  if (buffers_[local_replica_id]->allocated()) return;  // memory has already been allocated.
  buffers_[local_replica_id]->allocate();

  if (host_buffers_[local_replica_id]->allocated()) return;
  host_buffers_[local_replica_id]->allocate();
}

size_t EmbeddingManager::get_replica_batch_size() const {
  return global_batch_size_ / resource_mgr_->get_global_gpu_count();
}

void EmbeddingManager::get_output_shape(const std::shared_ptr<EmbeddingLayer>& emb,
                                        std::vector<int64_t>& output_shape,
                                        const bool dynamic_input) const {
  output_shape.clear();

  if (!dynamic_input) {                                // static input.shape
    output_shape.push_back(get_replica_batch_size());  // dim-0 is replica_batch_size
  }
  return emb->get_output_shape(output_shape, dynamic_input);
}

void EmbeddingManager::get_grad_shape(const size_t global_replica_id,
                                      const std::shared_ptr<EmbeddingLayer>& emb,
                                      std::vector<int64_t>& grad_shape) {
  const Context_t& replica_context = get_context(emb, global_replica_id);

  grad_shape.clear();
  return emb->get_grad_shape(replica_context, grad_shape);
}

void EmbeddingManager::forward(std::shared_ptr<EmbeddingLayer>& embedding,
                               const std::shared_ptr<Tensor> values,
                               const std::shared_ptr<Tensor> indices,
                               const size_t global_replica_id, const bool training,
                               std::shared_ptr<Tensor> embedding_vector,
                               std::shared_ptr<Tensor> h_replica_nnz) {
  Context_t& replica_context = get_context(embedding, global_replica_id);

  replica_context->set_input("replica_values", values, /*overwrite=*/true);
  replica_context->set_input("replica_indices", indices, /*overwrite=*/true);
  replica_context->set_output("replica_output", embedding_vector, /*overwrite=*/true);
  replica_context->set_output("replica_host_nnz", h_replica_nnz, /*overwrite=*/true);

  embedding->forward(replica_context, training);
}

void EmbeddingManager::forward(std::shared_ptr<EmbeddingLayer>& embedding,
                               const std::shared_ptr<Tensor> values, const size_t global_replica_id,
                               const bool training, std::shared_ptr<Tensor> embedding_vector,
                               std::shared_ptr<Tensor> h_replica_nnz) {
  Context_t& replica_context = get_context(embedding, global_replica_id);

  replica_context->set_input("replica_values", values, /*overwrite=*/true);
  replica_context->set_output("replica_output", embedding_vector, /*overwrite=*/true);
  replica_context->set_output("replica_host_nnz", h_replica_nnz, /*overwrite=*/true);

  embedding->forward(replica_context, training);
}

void EmbeddingManager::backward(std::shared_ptr<EmbeddingLayer>& embedding,
                                const std::shared_ptr<Tensor> top_gradient,
                                const size_t global_replica_id, std::shared_ptr<Tensor> gradient,
                                std::shared_ptr<Tensor> value_index) {
  Context_t& replica_context = get_context(embedding, global_replica_id);

  replica_context->set_input("replica_top_gradient", top_gradient, /*overwrite=*/true);
  replica_context->set_output("replica_input_grad", gradient, /*overwrite=*/true);
  replica_context->set_output("value_index_tensor", value_index, /*overwrite=*/true);

  embedding->backward(replica_context);
}

std::unordered_map<std::string, std::shared_ptr<UpdatePreparer>>
EmbeddingManager::create_preparers_for_Adam() {
  std::unordered_map<std::string, std::shared_ptr<UpdatePreparer>> preparers;

  // iterate on embeddings
  for (const auto& embedding : embeddings_) {
    const size_t global_batch_size = embedding->get_global_batch_size();
    const size_t max_feature_num = embedding->get_max_feature_num();
    const std::string var_name = embedding->get_var_name();
    const size_t max_vocabulary_size_per_gpu = embedding->get_max_vocabulary_size_per_gpu();

    auto preparer =
        GradUpdatePreparer::create(global_batch_size, max_feature_num, max_vocabulary_size_per_gpu,
                                   buffers_, host_buffers_, resource_mgr_);
    preparers.emplace(std::make_pair(var_name, preparer));
  }

  return preparers;
}

}  // namespace SparseOperationKit