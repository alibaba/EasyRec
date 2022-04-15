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

#ifndef OPERATION_HELPER_H
#define OPERATION_HELPER_H

#include <memory>
#include <mutex>
#include <string>

#include "common.h"
#include "dispatcher/dispatcher_builder.h"
#include "embeddings/embedding_lookuper_builder.h"
#include "operation/builder_container.h"

namespace SparseOperationKit {

class Registry {
 public:
  static Registry* instance();
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;
  Registry(Registry&&) = delete;
  Registry& operator=(Registry&&) = delete;

  template <typename DispatcherClass>
  int register_input_builder_helper(const OperationIdentifier op_identifier) {
    auto temp = std::shared_ptr<Builder>(new InputDispatcherBuilder<DispatcherClass>());
    std::lock_guard<std::mutex> lock(mu_);
    InputContainer::instance("input_dispatcher_builders")->push_back(op_identifier, temp);
    return 0;
  }

  template <typename OperationClass>
  int register_operation_builder_helper(const OperationIdentifier op_identifier) {
    auto temp = std::shared_ptr<Builder>(new OperationBuilder<OperationClass>());
    std::lock_guard<std::mutex> lock(mu_);
    OperationContainer::instance("operation_builders")->push_back(op_identifier, temp);
    return 0;
  }

  template <typename DispatcherClass>
  int register_output_builder_helper(const OperationIdentifier op_identifier) {
    auto temp = std::shared_ptr<Builder>(new OutputDispatcherBuilder<DispatcherClass>());
    std::lock_guard<std::mutex> lock(mu_);
    OutputContainer::instance("output_dispatcher_builders")->push_back(op_identifier, temp);
    return 0;
  }

  template <typename EmbeddingLookuperClass>
  int register_emb_lookuper_helper(const OperationIdentifier op_identifier) {
    auto temp = std::shared_ptr<Builder>(new EmbeddingLookuperBuilder<EmbeddingLookuperClass>());
    std::lock_guard<std::mutex> lock(mu_);
    LookuperContainer::instance("embedding_lookuper_builders")->push_back(op_identifier, temp);
    return 0;
  }

 private:
  Registry();
  std::mutex mu_;
};

#define UNIQUE_NAME_IMPL_2(x, y) x##y
#define UNIQUE_NAME_IMPL(x, y) UNIQUE_NAME_IMPL_2(x, y)
#define UNIQUE_NAME(x) UNIQUE_NAME_IMPL(x, __COUNTER__)

#define REGISTER_INPUT_DISPATCHER_BUILDER(name, key_dtype, dtype, ...) \
  static auto UNIQUE_NAME(in_dispatcher) =                             \
      Registry::instance()->register_input_builder_helper<__VA_ARGS__>({(name), (key_dtype), (dtype)});

#define REGISTER_OPERATION_BUILDER(name, key_dtype, dtype, ...) \
  static auto UNIQUE_NAME(operation) =                          \
      Registry::instance()->register_operation_builder_helper<__VA_ARGS__>({(name), (key_dtype), (dtype)});

#define REGISTER_OUTPUT_DISPATHER_BUILDER(name, key_dtype, dtype, ...) \
  static auto UNIQUE_NAME(out_dispatcher) =                            \
      Registry::instance()->register_output_builder_helper<__VA_ARGS__>({(name), (key_dtype), (dtype)});

#define REGISTER_EMB_LOOKUPER_BUILDER(name, key_dtype, dtype, ...) \
  static auto UNIQUE_NAME(lookuper) =                              \
      Registry::instance()->register_emb_lookuper_helper<__VA_ARGS__>({(name), (key_dtype), (dtype)});

}  // namespace SparseOperationKit

#endif  // OPERATION_HELPER_H