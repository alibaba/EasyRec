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

#ifndef TENSOR2_WRAPPER_H
#define TENSOR2_WRAPPER_H

#include "tensor_buffer/tensor2.hpp"
#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {

template <typename T>
class Tensor2Wrapper : public Tensor {
 public:
  static std::shared_ptr<Tensor2Wrapper> create(HugeCTR::Tensor2<T>& tensor2) {
    return std::shared_ptr<Tensor2Wrapper>(new Tensor2Wrapper(tensor2));
  }

  static std::vector<std::shared_ptr<Tensor2Wrapper>> create_many(
      std::vector<HugeCTR::Tensor2<T>> tensor2s) {
    std::vector<std::shared_ptr<Tensor2Wrapper>> result;
    for (auto iter = tensor2s.begin(); iter != tensor2s.end(); ++iter) {
      result.push_back(Tensor2Wrapper::create(*iter));
    }
    return result;
  }

  size_t get_size_in_bytes() override { return tensor2_.get_size_in_bytes(); }
  size_t get_num_elements() override { return tensor2_.get_num_elements(); }

  bool allocated() const override { return tensor2_.allocated(); }

  DataType dtype() const override { return dtype_; }

  void* get_ptr() override { return tensor2_.get_ptr(); }

 private:
  Tensor2Wrapper(HugeCTR::Tensor2<T>& tensor2) : tensor2_(tensor2) {}

  HugeCTR::Tensor2<T> tensor2_;
  const DataType dtype_ = DType<T>();
};

}  // namespace SparseOperationKit

#endif  // TENSOR2_WRAPPER_H