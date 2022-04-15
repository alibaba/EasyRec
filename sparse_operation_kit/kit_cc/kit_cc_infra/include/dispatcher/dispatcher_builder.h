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

#ifndef INPUT_DISPATCHER_BUILDER_H
#define INPUT_DISPATCHER_BUILDER_H

#include "operation/operation_builder.h"

namespace SparseOperationKit {

template <typename InputDispatcher>
class InputDispatcherBuilder : public Builder {
 public:
  InputDispatcherBuilder() {}
  std::shared_ptr<Operation> produce(ConstructionContext_t context) override {
    return std::make_shared<InputDispatcher>(context);
  }

  std::shared_ptr<EmbeddingLookuper> produce(ConstructionContext_t context,
                                             std::shared_ptr<ParamInterface> param) override {
    throw std::runtime_error(ErrorBase + "Not implemented.");
  }
};

template <typename OutputDispatcher>
class OutputDispatcherBuilder : public Builder {
 public:
  OutputDispatcherBuilder() {}
  std::shared_ptr<Operation> produce(ConstructionContext_t context) override {
    return std::make_shared<OutputDispatcher>(context);
  }
  std::shared_ptr<EmbeddingLookuper> produce(ConstructionContext_t context,
                                             std::shared_ptr<ParamInterface> param) override {
    throw std::runtime_error(ErrorBase + "Not implemented.");
  }
};

}  // namespace SparseOperationKit

#endif  // SparseOperationKit