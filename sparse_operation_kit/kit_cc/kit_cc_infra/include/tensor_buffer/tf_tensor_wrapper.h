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

#ifndef TF_TENSOR_WRAPPER_H
#define TF_TENSOR_WRAPPER_H

#include <initializer_list>

#include "tensor_buffer/tensor_interface.h"
#include "tensorflow/core/framework/tensor.h"

namespace SparseOperationKit {

DataType get_datatype(const tensorflow::DataType dtype);

/*
 * This is the wrapper for TF tensor to Tensor.
 */
class TFTensorWrapper : public Tensor {
 public:
  static std::shared_ptr<TFTensorWrapper> create(tensorflow::Tensor* tf_tensor);
  static std::vector<std::shared_ptr<TFTensorWrapper>> create_many(
      std::initializer_list<tensorflow::Tensor*> tf_tensors);

  size_t get_size_in_bytes() override;
  size_t get_num_elements() override;
  bool allocated() const override;
  DataType dtype() const override;

  void* get_ptr() override;

 private:
  TFTensorWrapper(tensorflow::Tensor* tf_tensor);

  tensorflow::Tensor* tf_tensor_;
  mutable DataType dtype_;
};

}  // namespace SparseOperationKit

#endif  // TF_TENSOR_WRAPPER_H