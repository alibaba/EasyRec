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

#ifndef EMBEDDING_BUFFER_H
#define EMBEDDING_BUFFER_H

#include "tensor_buffer/tensor_interface.h"
#include "tensorflow/core/framework/tensor.h"

namespace SparseOperationKit {

class EmbeddingBuffer : public tensorflow::TensorBuffer {
 public:
  static std::shared_ptr<EmbeddingBuffer> create(std::shared_ptr<Tensor> tensor);

  ~EmbeddingBuffer() override;
  size_t size() const override;
  tensorflow::TensorBuffer* root_buffer() override;
  void FillAllocationDescription(tensorflow::AllocationDescription* proto) const override;
#if TF_VERSION_MAJOR == 2
  bool GetAllocatedBytes(size_t* out_bytes) const override;
#endif
  bool OwnsMemory() const override;

  explicit EmbeddingBuffer(std::shared_ptr<Tensor> tensor);
  EmbeddingBuffer();

 private:
  std::shared_ptr<Tensor> tensor_;
};

class EmbeddingBufferBuilder {
 public:
  static std::shared_ptr<EmbeddingBufferBuilder> create(std::shared_ptr<Tensor> tensor);

  ~EmbeddingBufferBuilder();

  void build_buffer();
  tensorflow::TensorBuffer* get_init_buffer();

 private:
  EmbeddingBufferBuilder(std::shared_ptr<Tensor> tensor);

  std::shared_ptr<Tensor> tensor_;
  std::shared_ptr<EmbeddingBuffer> buffer_;
  const bool
      already_allocated_;  // whether the tensor is already allocated when creating this builder
};

}  // namespace SparseOperationKit

#endif  // EMBEDDING_BUFFER_H