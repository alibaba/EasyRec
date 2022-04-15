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

#include "parameters/param_interface.h"
#include "parameters/raw_param.h"
#include "common.h"

namespace SparseOperationKit {

// helper function to create Param Implementation
std::shared_ptr<ParamInterface> ParamInterface::CreateParam(
                                  const ParamType param_type,
                                  const std::string& initializer,
                                  const bool use_hashtable,
                                  const std::vector<size_t> shape,
                                  const std::shared_ptr<ResourcesManager>& resource_mgr,
                                  const std::string var_name, 
                                  const bool trainable,
                                  const DataType key_dtype, 
                                  const DataType value_dtype) {
  std::shared_ptr<ParamInterface> param{nullptr};
  switch (param_type) {
    case ParamType::RawParam: {
      RawParamCtor_t raw_param_creater = GetRawParamCtor(key_dtype, value_dtype);
      param = raw_param_creater(initializer, use_hashtable, shape, resource_mgr, var_name, trainable);
      break;
    }
    default: {
      throw std::runtime_error(ErrorBase + "UnKnown Parameter type.");
    }
  } // switch block
  return param;
}


ParamInterface::ParamInterface(const size_t max_vocabulary_size_per_gpu, 
                               const size_t embedding_vec_size,
                               const bool trainable, const std::string var_name,
                               const DataType dtype, const DataType key_dtype) 
: max_vocabulary_size_per_gpu_(max_vocabulary_size_per_gpu),
embedding_vec_size_(embedding_vec_size),
trainable_(trainable), var_name_(var_name), 
dtype_(dtype), key_dtype_(key_dtype) {}

size_t ParamInterface::get_max_vocabulary_size_per_gpu() const {
  return max_vocabulary_size_per_gpu_;
}

size_t ParamInterface::get_embedding_vec_size() const {
  return embedding_vec_size_;
}

bool ParamInterface::trainable() const {
  return trainable_;
}

std::string ParamInterface::get_var_name() const {
  return var_name_;
}

DataType ParamInterface::dtype() const {
  return dtype_;
}

DataType ParamInterface::key_dtype() const {
  return key_dtype_;
}

void ParamInterface::set_user(std::shared_ptr<EmbeddingLayer>& embedding) {
  // It is not compulsory for the subclass to override this function.
  throw std::runtime_error(ErrorBase + "Not implemented.");
}

void ParamInterface::let_user_dump_to_file(const std::string filepath) {
  // by default, it does nothing.
}

void ParamInterface::let_user_restore_from_file(const std::string filepath) {
  // by default, it does nothing.
}

void ParamInterface::let_user_load_embedding_values(
    const std::shared_ptr<Tensor>& tensor_list) {
  // by default, it does nothing
}

std::shared_ptr<Tensor>& ParamInterface::get_tensor(const size_t local_replica_id) {
  return get_embedding_table_tensor(local_replica_id);
}

void ParamInterface::set_hashtable(std::shared_ptr<BaseSimpleHashtable> hashtable) {
  throw std::runtime_error(ErrorBase + "Not implemented.");
}


TypeIdentity::TypeIdentity(const DataType key_dtype, const DataType value_dtype)
: key_dtype_(key_dtype), value_dtype_(value_dtype) {}

size_t TypeIdentityHash::operator()(const TypeIdentity& type_id) const {
  return std::hash<DataType>()(type_id.key_dtype_) ^ 
        (std::hash<DataType>()(type_id.value_dtype_) << 1);
}

bool TypeIdentityEqual::operator()(const TypeIdentity& lid, const TypeIdentity& rid) const {
  return (lid.key_dtype_ == rid.key_dtype_) && (lid.value_dtype_ == rid.value_dtype_);
}

}  // namespace SparseOperationKit