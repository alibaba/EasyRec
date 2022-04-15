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

#ifndef BUILDER_CONTAINER_H
#define BUILDER_CONTAINER_H

#include <string>
#include <unordered_map>

#include "operation/operation_builder.h"

namespace SparseOperationKit {

struct OperationIdentifier {
  OperationIdentifier(const std::string op_name, const DataType key_dtype, const DataType dtype);
  std::string DebugString() const;
  const std::string op_name_;
  const DataType key_dtype_;
  const DataType dtype_;
};

struct IdentifierHash {
  size_t operator()(const OperationIdentifier& identifier) const;
};

struct IdentifierEqual {
  bool operator()(const OperationIdentifier& lid, const OperationIdentifier& rid) const;
};

class BuilderContainer {
 public:
  explicit BuilderContainer(const std::string name);
  void push_back(const OperationIdentifier op_identifier, const std::shared_ptr<Builder> builder);
  std::shared_ptr<Builder> get_builder(const OperationIdentifier op_identifier);
  std::vector<std::string> get_builder_names() const;

 protected:
  const std::string name_;
  std::unordered_map<OperationIdentifier, std::shared_ptr<Builder>, 
                    IdentifierHash, IdentifierEqual> components_;
};

class InputContainer : public BuilderContainer {
 public:
  static InputContainer* instance(const std::string name);
  explicit InputContainer(const std::string name);
};

class OutputContainer : public BuilderContainer {
 public:
  static OutputContainer* instance(const std::string name);
  explicit OutputContainer(const std::string name);
};

class OperationContainer : public BuilderContainer {
 public:
  static OperationContainer* instance(const std::string name);
  explicit OperationContainer(const std::string name);
};

class LookuperContainer : public BuilderContainer {
 public:
  static LookuperContainer* instance(const std::string name);
  explicit LookuperContainer(const std::string name);
};

}  // namespace SparseOperationKit

#endif  // BUILDER_CONTAINER_H