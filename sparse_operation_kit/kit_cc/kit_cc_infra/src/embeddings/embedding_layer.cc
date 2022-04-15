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

#include "embeddings/embedding_layer.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace SparseOperationKit {

EmbeddingLayer::EmbeddingLayer(std::shared_ptr<Dispatcher> input_dispatcher,
                               std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
                               std::shared_ptr<Dispatcher> output_dispatcher,
                               ConstructionContext_t context)
    : input_dispatcher_(input_dispatcher),
      embedding_lookuper_(embedding_lookuper),
      output_dispatcher_(output_dispatcher),
      base_context_(context),
      global_batch_size_(context->get_replica_batch_size() *
                         context->get_resource_mgr()->get_global_gpu_count()) {}

std::shared_ptr<EmbeddingLayer> EmbeddingLayer::create(
    std::shared_ptr<Dispatcher> input_dispatcher,
    std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
    std::shared_ptr<Dispatcher> output_dispatcher, ConstructionContext_t context) {
  return std::shared_ptr<EmbeddingLayer>(
      new EmbeddingLayer(input_dispatcher, embedding_lookuper, output_dispatcher, context));
}

void EmbeddingLayer::allocate_forward_spaces() {
  input_dispatcher_->AllocateForwardSpaces();
  embedding_lookuper_->AllocateForwardSpaces();
  output_dispatcher_->AllocateForwardSpaces();
}

void EmbeddingLayer::allocate_backward_spaces() {
  input_dispatcher_->AllocateBackwardSpaces();
  embedding_lookuper_->AllocateBackwardSpaces();
  output_dispatcher_->AllocateBackwardSpaces();
}

void EmbeddingLayer::forward(const Context_t& replica_context, const bool training) {
  // step 1 dispatch input to each GPU, Data-Parallel -> Model Parallel
  input_dispatcher_->Forward(replica_context, training);

  // step 2 do embedding lookup on each GPU independently
  embedding_lookuper_->Forward(replica_context, training);

  // step 3 dispatch embedding vector to each GPU, Model-Parallel -> Data-Parallel
  output_dispatcher_->Forward(replica_context, training);
}

void EmbeddingLayer::backward(const Context_t& replica_context) {
  // step 1 dispatch top_gradients to each GPU, Data-Parallel -> Model Parallel
  output_dispatcher_->Backward(replica_context);

  // step 2 do backward on each GPU independently
  embedding_lookuper_->Backward(replica_context);

  // step 3 dispatch input grads to each GPU, Model-Parallel -> Data Parallel
  input_dispatcher_->Backward(replica_context);
}

void EmbeddingLayer::get_output_shape(std::vector<int64_t>& output_shape,
                                      const bool dynamic_input) const {
  // for sparse embedding layer, dynamic_input is ignored.
  // dim-0 is replica_batch_size, which is already set by EmbeddingManager
  output_shape.push_back(base_context_->get_slot_num());
  output_shape.push_back(base_context_->get_param()->get_embedding_vec_size());

  // check its rank
  if (3 != output_shape.size())
    throw std::runtime_error(ErrorBase + "For Sparse Embedding Layer, the output shape " +
                             "should be [replica_batch_size, slot_num, emb_vec_size]. " +
                             "But now its rank is " + std::to_string(output_shape.size()));
}

void EmbeddingLayer::get_grad_shape(const Context_t& replica_context,
                                    std::vector<int64_t>& grad_shape) const {
  // replica_host_nnz is set by TF, therefore only append embedding_vec_size.
  grad_shape.push_back(static_cast<int64_t>(base_context_->get_param()->get_embedding_vec_size()));
}

size_t EmbeddingLayer::get_global_batch_size() const { return global_batch_size_; }

size_t EmbeddingLayer::get_max_feature_num() const { return base_context_->get_max_feature_num(); }

std::string EmbeddingLayer::get_var_name() const {
  return base_context_->get_param()->get_var_name();
}

size_t EmbeddingLayer::get_max_vocabulary_size_per_gpu() const {
  return base_context_->get_param()->get_max_vocabulary_size_per_gpu();
}

void EmbeddingLayer::dump_to_file(const std::string filepath) const {
  input_dispatcher_->DumpToFile(filepath);

  embedding_lookuper_->DumpToFile(filepath);

  output_dispatcher_->DumpToFile(filepath);
}

void EmbeddingLayer::restore_from_file(const std::string filepath) {
  input_dispatcher_->RestoreFromFile(filepath);

  embedding_lookuper_->RestoreFromFile(filepath);

  output_dispatcher_->RestoreFromFile(filepath);
}

void EmbeddingLayer::restore_params(const std::shared_ptr<Tensor>& keys,
                                    const std::shared_ptr<Tensor>& embedding_values,
                                    const size_t num_total_keys) {
  // because the params is only used by embedding lookuper,
  // so that delegate this job to embedding lookuper
  embedding_lookuper_->restore_params(keys, embedding_values, num_total_keys);
}

void EmbeddingLayer::save_params(std::shared_ptr<Tensor>& keys,
                                 std::shared_ptr<Tensor>& embedding_values,
                                 size_t& num_total_keys) const {
  // because the params is only used by embedding lookuper,
  // so that delegate this job to embedding lookuper
  embedding_lookuper_->save_params(keys, embedding_values, num_total_keys);
}

void EmbeddingLayer::load_embedding_values(
    const std::shared_ptr<Tensor>& emb_values) {
  input_dispatcher_->LoadEmbeddingValues(emb_values);

  embedding_lookuper_->LoadEmbeddingValues(emb_values);

  output_dispatcher_->LoadEmbeddingValues(emb_values);
}

DataType EmbeddingLayer::key_dtype() const {
  return base_context_->key_dtype();
}

DataType EmbeddingLayer::compute_dtype() const {
  return base_context_->compute_dtype();
}

ConstructionContext_t EmbeddingLayer::base_context() const { return base_context_; }

DenseEmbeddingLayer::DenseEmbeddingLayer(std::shared_ptr<Dispatcher> input_dispatcher,
                                         std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
                                         std::shared_ptr<Dispatcher> output_dispatcher,
                                         ConstructionContext_t context)
    : EmbeddingLayer(input_dispatcher, embedding_lookuper, output_dispatcher, context) {}

std::shared_ptr<DenseEmbeddingLayer> DenseEmbeddingLayer::create(
    std::shared_ptr<Dispatcher> input_dispatcher,
    std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
    std::shared_ptr<Dispatcher> output_dispatcher, ConstructionContext_t context) {
  return std::shared_ptr<DenseEmbeddingLayer>(
      new DenseEmbeddingLayer(input_dispatcher, embedding_lookuper, output_dispatcher, context));
}

void DenseEmbeddingLayer::get_output_shape(std::vector<int64_t>& output_shape,
                                           const bool dynamic_input) const {
  if (dynamic_input) {  // input.shape is dynamic
    // only emb_vec_size is set here, None will be set by the input.shape
    output_shape.push_back(base_context()->get_param()->get_embedding_vec_size());
    if (1 != output_shape.size())
      throw std::runtime_error(ErrorBase +
                               "For DenseEmbeddingLayer, when dynamic_input is specified, " +
                               "the output shape should be [None, emb_vec_size].");
  } else {  // input.shape is static
    // dim-0 is replica_batch_size, which is already set by EmbeddingManager
    output_shape.push_back(base_context()->get_slot_num());
    output_shape.push_back(base_context()->get_nnz_per_slot());
    output_shape.push_back(base_context()->get_param()->get_embedding_vec_size());

    // check its rank
    if (4 != output_shape.size())
      throw std::runtime_error(
          ErrorBase + "For Dense Embedding Layer, the output shape " +
          "should be [replica_batch_size, slot_num, nnz_per_slot, emb_vec_size]. " +
          "But now its rank is " + std::to_string(output_shape.size()));
  }
}

size_t DenseEmbeddingLayer::get_max_feature_num() const {
  return base_context()->get_slot_num() * base_context()->get_nnz_per_slot();
}

class EmbeddingVariantWrapper {
 public:
  EmbeddingVariantWrapper() : embedding_(nullptr) {}
  explicit EmbeddingVariantWrapper(const std::shared_ptr<EmbeddingLayer> emb) : embedding_(emb) {}
  EmbeddingVariantWrapper(const EmbeddingVariantWrapper& other) : embedding_(other.embedding_) {}
  EmbeddingVariantWrapper& operator=(EmbeddingVariantWrapper&& other) {
    if (&other == this) return *this;
    embedding_ = other.embedding_;
    return *this;
  }
  EmbeddingVariantWrapper& operator=(const EmbeddingVariantWrapper& other) = delete;

  std::shared_ptr<EmbeddingLayer> get() const { return embedding_; }

  ~EmbeddingVariantWrapper() = default;
  tensorflow::string TypeName() const { return "EmbeddingPlugin::EmbeddingVariantWrapper"; }
  void Encode(tensorflow::VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "EmbeddingVariantWrapper objects.";
  }
  bool Decode(const tensorflow::VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "EmbeddingVariantWrapper objects.";
    return false;
  }

 private:
  std::shared_ptr<EmbeddingLayer> embedding_;
};

void GetEmbeddingFromVariantTensor(const tensorflow::Tensor* tensor,
                                   std::shared_ptr<EmbeddingLayer>& out_emb) {
  if (!(tensor->dtype() == tensorflow::DT_VARIANT &&
        tensorflow::TensorShapeUtils::IsScalar(tensor->shape()))) {
    throw std::runtime_error(ErrorBase + "Embedding tensor must be a scalar of dtype DT_VARIANT.");
  }
  const tensorflow::Variant& variant = tensor->scalar<tensorflow::Variant>()();
  const EmbeddingVariantWrapper* wrapper = variant.get<EmbeddingVariantWrapper>();
  if (nullptr == wrapper)
    throw std::runtime_error(ErrorBase + "Tensor must be a EmbeddingPlugin::Embedding object.");
  out_emb = wrapper->get();
  if (!out_emb) throw std::runtime_error(ErrorBase + "read empty embedding object.");
}

void StoreEmbeddingInVariantTensor(const std::shared_ptr<EmbeddingLayer>& emb,
                                   tensorflow::Tensor* tensor) {
  if (!(tensor->dtype() == tensorflow::DT_VARIANT &&
        tensorflow::TensorShapeUtils::IsScalar(tensor->shape()))) {
    throw std::runtime_error(ErrorBase + "Embedding tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<tensorflow::Variant>()() = EmbeddingVariantWrapper(emb);
}

}  // namespace SparseOperationKit