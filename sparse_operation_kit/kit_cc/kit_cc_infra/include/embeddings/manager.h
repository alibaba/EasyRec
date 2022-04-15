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

#ifndef EMBEDDINGS_MANAGER_H
#define EMBEDDINGS_MANAGER_H

#include <memory>
#include <mutex>

#include "embeddings/embedding_layer.h"
#include "operation/op_context.h"
#include "optimizer/update_preparer.h"
#include "parameters/param_interface.h"
#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"
#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {

class EmbeddingManager final {
 public:
  ~EmbeddingManager() = default;
  EmbeddingManager(const EmbeddingManager&) = delete;
  EmbeddingManager& operator=(const EmbeddingManager&) = delete;

  static std::shared_ptr<EmbeddingManager> Create(
      const std::shared_ptr<ResourcesManager>& resource_mgr);
  void init(const size_t global_replica_id, const size_t global_batch_size);

  // create sparse embedding layer
  void create_embedding(const std::shared_ptr<ParamInterface>& param,
                        const std::string input_dispatcher,
                        const std::vector<std::string> input_dispatcher_subsequent_ops,
                        const std::string embedding_executor, const std::string output_dispatcher,
                        const std::vector<std::string> output_dispatcher_subsequent_ops,
                        const size_t slot_num, const size_t max_nnz, const size_t max_feature_num,
                        const CombinerType combiner, const DataType compute_dtype,
                        std::shared_ptr<EmbeddingLayer>& embedding);

  // create dense embedding layer
  void create_embedding(const std::shared_ptr<ParamInterface>& param,
                        const std::string input_dispatcher,
                        const std::vector<std::string> input_dispatcher_subsequent_ops,
                        const std::string embedding_lookuper, const std::string output_dispatcher,
                        const std::vector<std::string> output_dispatcher_subsequent_ops,
                        const size_t slot_num, const size_t nnz_per_slot,
                        const DataType compute_dtype,
                        std::shared_ptr<EmbeddingLayer>& embedding);

  void allocate_memory(const size_t global_replica_id);

  void get_output_shape(const std::shared_ptr<EmbeddingLayer>& emb,
                        std::vector<int64_t>& output_shape, const bool dynamic_input) const;

  void get_grad_shape(const size_t global_replica_id, const std::shared_ptr<EmbeddingLayer>& emb,
                      std::vector<int64_t>& grad_shape);

  void forward(std::shared_ptr<EmbeddingLayer>& embedding, const std::shared_ptr<Tensor> values,
               const std::shared_ptr<Tensor> indices, const size_t global_replica_id,
               const bool training, std::shared_ptr<Tensor> embedding_vector,
               std::shared_ptr<Tensor> h_replica_nnz);

  void forward(std::shared_ptr<EmbeddingLayer>& embedding, const std::shared_ptr<Tensor> values,
               const size_t global_replica_id, const bool training,
               std::shared_ptr<Tensor> embedding_vector,
               std::shared_ptr<Tensor> h_replica_nnz);

  void backward(std::shared_ptr<EmbeddingLayer>& embedding,
                const std::shared_ptr<Tensor> top_gradient, const size_t global_replica_id,
                std::shared_ptr<Tensor> gradient, std::shared_ptr<Tensor> value_index);

  std::unordered_map<std::string, std::shared_ptr<UpdatePreparer>> create_preparers_for_Adam();

 private:
  EmbeddingManager(const std::shared_ptr<ResourcesManager>& resource_mgr);
  size_t get_replica_batch_size() const;
  void create_contexts(std::shared_ptr<EmbeddingLayer> embedding);
  Context_t& get_context(const std::shared_ptr<EmbeddingLayer>& embedding,
                         const size_t global_replica_id);

  std::shared_ptr<ResourcesManager> resource_mgr_;

  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>
      buffers_;  // used to save internal spaces for all embedding.
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>>> host_buffers_;
  size_t global_batch_size_;
  volatile bool resized_;

  std::vector<std::shared_ptr<EmbeddingLayer>> embeddings_;  // store all embedding layers.
  std::unordered_map<std::shared_ptr<EmbeddingLayer>, std::vector<Context_t>> embedding_contexts_;

  std::mutex mu_;
};

}  // namespace SparseOperationKit

#endif  // EMBEDDINGS_MANAGER_H