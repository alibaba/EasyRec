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

#ifndef EMBEDDING_LAYER_H
#define EMBEDDING_LAYER_H

#include "embeddings/embedding_lookuper.h"
#include "operation/operation.h"
#include "tensorflow/core/framework/tensor.h"

namespace SparseOperationKit {

class EmbeddingLayer {
 public:
  static std::shared_ptr<EmbeddingLayer> create(
      std::shared_ptr<Dispatcher> input_dispatcher,
      std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
      std::shared_ptr<Dispatcher> output_dispatcher, 
      ConstructionContext_t context);

  void allocate_forward_spaces();
  void allocate_backward_spaces();
  void forward(const Context_t &replica_context, const bool training);
  void backward(const Context_t &replica_context);

  virtual void get_output_shape(std::vector<int64_t> &output_shape,
                                const bool dynamic_input = false) const;
  void get_grad_shape(const Context_t &replica_context, std::vector<int64_t> &grad_shape) const;

  size_t get_global_batch_size() const;
  virtual size_t get_max_feature_num() const;
  std::string get_var_name() const;
  size_t get_max_vocabulary_size_per_gpu() const;

  void dump_to_file(const std::string filepath) const;
  // restore the operations' content from file, except trainable embedding variable's
  void restore_from_file(const std::string filepath);
  // help to restore params
  void restore_params(const std::shared_ptr<Tensor> &keys,
                      const std::shared_ptr<Tensor> &embedding_values, const size_t num_total_keys);
  // help to save params
  void save_params(std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
                   size_t &num_total_keys) const;
  void load_embedding_values(const std::shared_ptr<Tensor> &emb_values);

  DataType key_dtype() const;
  DataType compute_dtype() const;


 protected:
  EmbeddingLayer(std::shared_ptr<Dispatcher> input_dispatcher,
                 std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
                 std::shared_ptr<Dispatcher> output_dispatcher, ConstructionContext_t context);

  ConstructionContext_t base_context() const;

 private:
  const std::shared_ptr<Dispatcher> input_dispatcher_;
  const std::shared_ptr<EmbeddingLookuper> embedding_lookuper_;
  const std::shared_ptr<Dispatcher> output_dispatcher_;
  const ConstructionContext_t base_context_;
  const size_t global_batch_size_;
};

class DenseEmbeddingLayer : public EmbeddingLayer {
 public:
  static std::shared_ptr<DenseEmbeddingLayer> create(
      std::shared_ptr<Dispatcher> input_dispatcher,
      std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
      std::shared_ptr<Dispatcher> output_dispatcher, ConstructionContext_t context);

  void get_output_shape(std::vector<int64_t> &output_shape,
                        const bool dynamic_input = false) const override;
  size_t get_max_feature_num() const override;

 private:
  DenseEmbeddingLayer(std::shared_ptr<Dispatcher> input_dispatcher,
                      std::shared_ptr<EmbeddingLookuper> embedding_lookuper,
                      std::shared_ptr<Dispatcher> output_dispatcher, ConstructionContext_t context);
};

void GetEmbeddingFromVariantTensor(const tensorflow::Tensor *tensor,
                                   std::shared_ptr<EmbeddingLayer> &out_emb);
void StoreEmbeddingInVariantTensor(const std::shared_ptr<EmbeddingLayer> &emb,
                                   tensorflow::Tensor *tensor);

}  // namespace SparseOperationKit

#endif  // EMBEDDING_LAYER_H