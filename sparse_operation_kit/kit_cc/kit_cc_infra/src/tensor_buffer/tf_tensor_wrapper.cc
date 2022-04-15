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

#include "tensor_buffer/tf_tensor_wrapper.h"

namespace SparseOperationKit {

DataType get_datatype(const tensorflow::DataType dtype) {
  switch (dtype) {
    case tensorflow::DataType::DT_FLOAT: { return DataType::Float32; }
    case tensorflow::DataType::DT_HALF: { return DataType::Float16; }
    case tensorflow::DataType::DT_INT64: { return DataType::Int64; }
    case tensorflow::DataType::DT_UINT64: { return DataType::Uint64; }
    case tensorflow::DataType::DT_INT32: { return DataType::Int32; }
    case tensorflow::DataType::DT_UINT32: { return DataType::Uint32; }
    default: {
      throw std::runtime_error(ErrorBase + "Not supported dtype.");
      break;
    }
  } // switch block
}

size_t size_of(tensorflow::DataType data_type) {
  switch (data_type) {
    case tensorflow::DataType::DT_UINT8:
    case tensorflow::DataType::DT_INT8:
    case tensorflow::DataType::DT_BOOL:
      return 1;
    case tensorflow::DataType::DT_HALF:
    case tensorflow::DataType::DT_INT16:
    case tensorflow::DataType::DT_UINT16:
      return 2;
    case tensorflow::DataType::DT_FLOAT:
    case tensorflow::DataType::DT_INT32:
    case tensorflow::DataType::DT_UINT32:
      return 4;
    case tensorflow::DataType::DT_DOUBLE:
    case tensorflow::DataType::DT_INT64:
    case tensorflow::DataType::DT_UINT64:
      return 8;
    default:
      return 0;
  }  // switch data_type
  return 0;
}

TFTensorWrapper::TFTensorWrapper(tensorflow::Tensor* tf_tensor) 
: tf_tensor_(tf_tensor), dtype_(DataType::Unknown) {}

std::shared_ptr<TFTensorWrapper> TFTensorWrapper::create(tensorflow::Tensor* tf_tensor) {
  return std::shared_ptr<TFTensorWrapper>(new TFTensorWrapper(tf_tensor));
}

std::vector<std::shared_ptr<TFTensorWrapper>> TFTensorWrapper::create_many(
    std::initializer_list<tensorflow::Tensor*> tf_tensors) {
  std::vector<std::shared_ptr<TFTensorWrapper>> result;
  for (auto iter = tf_tensors.begin(); iter != tf_tensors.end(); ++iter) {
    result.push_back(create(*iter));
  }
  return result;
}

void* TFTensorWrapper::get_ptr() { return tf_tensor_->data(); }

size_t TFTensorWrapper::get_size_in_bytes() {
  return get_num_elements() * size_of(tf_tensor_->dtype());
}

size_t TFTensorWrapper::get_num_elements() {
  return static_cast<size_t>(tf_tensor_->NumElements());
}

bool TFTensorWrapper::allocated() const { return tf_tensor_->IsInitialized(); }

DataType TFTensorWrapper::dtype() const { 
  if (DataType::Unknown == dtype_) { dtype_ = get_datatype(tf_tensor_->dtype()); }
  return dtype_; 
}

}  // namespace SparseOperationKit