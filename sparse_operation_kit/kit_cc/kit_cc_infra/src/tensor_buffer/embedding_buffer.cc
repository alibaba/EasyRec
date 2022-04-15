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

#include "tensor_buffer/embedding_buffer.h"

#include "common.h"

namespace SparseOperationKit {

std::shared_ptr<EmbeddingBuffer> EmbeddingBuffer::create(std::shared_ptr<Tensor> tensor) {
  auto Deleter = [](EmbeddingBuffer* buf) {
    // SOK do not delete this object, TF will delete it.
  };
  if (tensor)
    return std::shared_ptr<EmbeddingBuffer>(new EmbeddingBuffer(tensor), Deleter);
  else
    return std::shared_ptr<EmbeddingBuffer>(new EmbeddingBuffer(), Deleter);
}

EmbeddingBuffer::EmbeddingBuffer() : tensorflow::TensorBuffer(nullptr), tensor_(nullptr) {}

EmbeddingBuffer::EmbeddingBuffer(std::shared_ptr<Tensor> tensor)
    : tensorflow::TensorBuffer(tensor->GetPtrWithType<float>()), tensor_(tensor) {}

EmbeddingBuffer::~EmbeddingBuffer() {}

size_t EmbeddingBuffer::size() const { return tensor_->get_size_in_bytes(); }

tensorflow::TensorBuffer* EmbeddingBuffer::root_buffer() { return this; }

void EmbeddingBuffer::FillAllocationDescription(tensorflow::AllocationDescription* proto) const {
  // TODO: need implementation
}

#if TF_VERSION_MAJOR == 2
bool EmbeddingBuffer::GetAllocatedBytes(size_t* out_bytes) const {
  *out_bytes = size();
  return *out_bytes > 0;
}
#endif

bool EmbeddingBuffer::OwnsMemory() const { return true; }

std::shared_ptr<EmbeddingBufferBuilder> EmbeddingBufferBuilder::create(
    std::shared_ptr<Tensor> tensor) {
  return std::shared_ptr<EmbeddingBufferBuilder>(new EmbeddingBufferBuilder(tensor));
}

EmbeddingBufferBuilder::EmbeddingBufferBuilder(std::shared_ptr<Tensor> tensor)
    : tensor_(tensor), buffer_(nullptr), already_allocated_(tensor->allocated()) {}

EmbeddingBufferBuilder::~EmbeddingBufferBuilder() {}

void EmbeddingBufferBuilder::build_buffer() {
  if (!already_allocated_) {  // only need to create a new buffer when the buffer is not allocated
                              // in the beginning.
    if (!tensor_->allocated())
      throw std::runtime_error(ErrorBase + "Have not allocated memory for tensor.");
    if (!buffer_)
      throw std::runtime_error(ErrorBase + "Have not allocate spaces for EmbeddingBuffer");

    // Release the old one and Construct a new EmbeddingBuffer in the existed space
    buffer_->~EmbeddingBuffer();
    new (buffer_.get()) EmbeddingBuffer(tensor_);
  } else {
    buffer_->Unref();
  }
}

tensorflow::TensorBuffer* EmbeddingBufferBuilder::get_init_buffer() {
  buffer_ =
      already_allocated_ ? EmbeddingBuffer::create(tensor_) : EmbeddingBuffer::create(nullptr);
  return buffer_.get();
}

}  // namespace SparseOperationKit