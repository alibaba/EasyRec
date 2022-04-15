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

#pragma once
#include <cuda_runtime_api.h>

#include <memory>

#include "tensor2.hpp"

namespace HugeCTR {

class HostAllocator {
 public:
  void *allocate(size_t size) const { return malloc(size); }
  void deallocate(void *ptr) const { free(ptr); }
};

class CudaHostAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA(cudaFreeHost(ptr)); }
};

class CudaManagedAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA(cudaMallocManaged(&ptr, size));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA(cudaFree(ptr)); }
};

class CudaAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA(cudaFree(ptr)); }
};

template <typename T>
class BufferBlock2 {
 public:
  virtual ~BufferBlock2() {}
  virtual void reserve(const std::vector<size_t> &dimensions, Tensor2<T> *tensor) = 0;
  virtual Tensor2<T> &as_tensor() = 0;
};

template <typename Allocator>
class GeneralBuffer2 : public std::enable_shared_from_this<GeneralBuffer2<Allocator>> {
  class BufferInternal {
   public:
    virtual ~BufferInternal() {}
    virtual size_t get_size_in_bytes() const = 0;
    virtual void initialize(const std::shared_ptr<GeneralBuffer2> &buffer, size_t offset) = 0;
  };

  class TensorBufferImpl : public TensorBuffer2, public BufferInternal {
    size_t size_in_bytes_;
    std::shared_ptr<GeneralBuffer2> buffer_;
    size_t offset_;

   public:
    TensorBufferImpl(size_t size_in_bytes) : size_in_bytes_(size_in_bytes) {}
    bool allocated() const override { return buffer_ && buffer_->allocated(); }
    void *get_ptr() override { return forward_void_pointer(buffer_->ptr_, offset_); }

    size_t get_size_in_bytes() const { return size_in_bytes_; }
    void initialize(const std::shared_ptr<GeneralBuffer2> &buffer, size_t offset) {
      buffer_ = buffer;
      offset_ = offset;
    }
  };

  template <typename T>
  class BufferBlockImpl : public BufferBlock2<T>, public BufferInternal {
    size_t total_num_elements_;
    std::shared_ptr<TensorBufferImpl> buffer_impl_;
    Tensor2<T> tensor_;
    bool finalized_;
    std::vector<std::shared_ptr<BufferInternal>> reserved_buffers_;

   public:
    BufferBlockImpl() : total_num_elements_(0), finalized_(false) {}

    void reserve(const std::vector<size_t> &dimensions, Tensor2<T> *tensor) override {
      if (finalized_) {
        throw std::runtime_error(ErrorBase + "Buffer block is finalized.");
      }
      size_t num_elements = get_num_elements_from_dimensions(dimensions);
      size_t size_in_bytes = num_elements * TensorScalarSizeFunc<T>::get_element_size();

      std::shared_ptr<TensorBufferImpl> buffer_impl =
          std::make_shared<TensorBufferImpl>(size_in_bytes);
      reserved_buffers_.push_back(buffer_impl);

      *tensor = Tensor2<T>(dimensions, buffer_impl);

      total_num_elements_ += num_elements;
    }

    Tensor2<T> &as_tensor() override {
      if (!finalized_) {
        buffer_impl_ = std::make_shared<TensorBufferImpl>(
            total_num_elements_ * TensorScalarSizeFunc<T>::get_element_size());
        tensor_ = Tensor2<T>({total_num_elements_}, buffer_impl_);
        finalized_ = true;
      }
      return tensor_;
    };

    size_t get_size_in_bytes() const {
      return total_num_elements_ * TensorScalarSizeFunc<T>::get_element_size();
    }

    void initialize(const std::shared_ptr<GeneralBuffer2> &buffer, size_t offset) {
      size_t local_offset = 0;
      for (const std::shared_ptr<BufferInternal> &buffer_impl : reserved_buffers_) {
        buffer_impl->initialize(buffer, offset + local_offset);
        local_offset += buffer_impl->get_size_in_bytes();
      }
      reserved_buffers_.clear();

      if (!finalized_) {
        buffer_impl_ = std::make_shared<TensorBufferImpl>(
            total_num_elements_ * TensorScalarSizeFunc<T>::get_element_size());
        tensor_ = Tensor2<T>({total_num_elements_}, buffer_impl_);
        finalized_ = true;
      }
      buffer_impl_->initialize(buffer, offset);
    }
  };

  Allocator allocator_;
  void *ptr_;
  size_t total_size_in_bytes_;
  std::vector<std::shared_ptr<BufferInternal>> reserved_buffers_;

  GeneralBuffer2() : ptr_(nullptr), total_size_in_bytes_(0) {}

 public:
  static std::shared_ptr<GeneralBuffer2> create() {
    return std::shared_ptr<GeneralBuffer2>(new GeneralBuffer2);
  }

  GeneralBuffer2(const GeneralBuffer2 &) = delete;
  GeneralBuffer2 &operator=(const GeneralBuffer2 &) = delete;

  ~GeneralBuffer2() {
    if (allocated()) {
      allocator_.deallocate(ptr_);
    }
  }

  void allocate() {
    if (ptr_ != nullptr) {
      throw std::runtime_error(ErrorBase + "Memory has already been allocated.");
    }

    size_t offset = 0;
    for (const std::shared_ptr<BufferInternal> &buffer : reserved_buffers_) {
      buffer->initialize(this->shared_from_this(), offset);
      size_t size_in_bytes = buffer->get_size_in_bytes();
      if (size_in_bytes % 32 != 0) {
        size_in_bytes += (32 - size_in_bytes % 32);
      }
      offset += size_in_bytes;
    }
    reserved_buffers_.clear();
    total_size_in_bytes_ = offset;

    if (total_size_in_bytes_ != 0) {
      ptr_ = allocator_.allocate(total_size_in_bytes_);
    }
  }

  template <typename T>
  std::shared_ptr<BufferBlock2<T>> create_block() {
    if (allocated()) {
      throw std::runtime_error(ErrorBase + "General buffer is finalized.");
    }
    std::shared_ptr<BufferBlockImpl<T>> block_impl = std::make_shared<BufferBlockImpl<T>>();
    reserved_buffers_.push_back(block_impl);
    return block_impl;
  }

  template <typename T>
  void reserve(const std::vector<size_t> &dimensions, Tensor2<T> *tensor) {
    if (allocated()) {
      throw std::runtime_error(ErrorBase + "General buffer is finalized.");
    }

    size_t size_in_bytes =
        get_num_elements_from_dimensions(dimensions) * TensorScalarSizeFunc<T>::get_element_size();

    std::shared_ptr<TensorBufferImpl> buffer_impl =
        std::make_shared<TensorBufferImpl>(size_in_bytes);
    reserved_buffers_.push_back(buffer_impl);

    *tensor = Tensor2<T>(dimensions, buffer_impl);
  }

  bool allocated() const { return total_size_in_bytes_ != 0 && ptr_ != nullptr; }

};  // namespace HugeCTR

}  // namespace HugeCTR