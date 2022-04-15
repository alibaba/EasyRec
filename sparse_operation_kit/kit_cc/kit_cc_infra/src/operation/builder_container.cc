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

#include "operation/builder_container.h"

#include "common.h"

namespace SparseOperationKit {

OperationIdentifier::OperationIdentifier(const std::string op_name, 
                                         const DataType key_dtype, 
                                         const DataType dtype)
: op_name_(op_name), key_dtype_(key_dtype), dtype_(dtype) {}

std::string OperationIdentifier::DebugString() const {
  return op_name_ + " with key_dtype = " + DataTypeString(key_dtype_) 
          + ", dtype = " + DataTypeString(dtype_);
}

size_t IdentifierHash::operator()(const OperationIdentifier& identifier) const {
  return std::hash<std::string>()(identifier.op_name_) ^
          (std::hash<DataType>()(identifier.key_dtype_) << 1) ^ 
          (std::hash<DataType>()(identifier.dtype_) << 2);
}

bool IdentifierEqual::operator()(const OperationIdentifier& lid, 
                                 const OperationIdentifier& rid) const {
  return (lid.op_name_ == rid.op_name_) && 
         (lid.key_dtype_ == rid.key_dtype_) && 
         (lid.dtype_ == rid.dtype_);
}

BuilderContainer::BuilderContainer(const std::string name) : name_(name) {}

void BuilderContainer::push_back(const OperationIdentifier op_identifier,
                                 const std::shared_ptr<Builder> builder) {
  auto iter = components_.find(op_identifier);
  if (components_.end() != iter)
    throw std::runtime_error(ErrorBase + "There already exists a builder "
                             + op_identifier.DebugString() + " in container: " + name_);

  components_.emplace(std::make_pair(op_identifier, builder));
}

std::shared_ptr<Builder> BuilderContainer::get_builder(const OperationIdentifier op_identifier) {
  auto iter = components_.find(op_identifier);
  if (components_.end() == iter)
    throw std::runtime_error(ErrorBase + "Cannot find " + op_identifier.DebugString() 
                             + " in container: " + name_);

  return iter->second;
}

std::vector<std::string> BuilderContainer::get_builder_names() const {
  std::vector<std::string> builder_names;
  for (auto iter : components_) {
    builder_names.emplace_back(iter.first.op_name_);
  }
  return builder_names;
}

InputContainer::InputContainer(const std::string name) : BuilderContainer(name) {}

InputContainer* InputContainer::instance(const std::string name) {
  static InputContainer instance(name);
  return &instance;
}

OutputContainer::OutputContainer(const std::string name) : BuilderContainer(name) {}

OutputContainer* OutputContainer::instance(const std::string name) {
  static OutputContainer instance(name);
  return &instance;
}

OperationContainer::OperationContainer(const std::string name) : BuilderContainer(name) {}

OperationContainer* OperationContainer::instance(const std::string name) {
  static OperationContainer instance(name);
  return &instance;
}

LookuperContainer::LookuperContainer(const std::string name) : BuilderContainer(name) {}

LookuperContainer* LookuperContainer::instance(const std::string name) {
  static LookuperContainer instance(name);
  return &instance;
}

}  // namespace SparseOperationKit