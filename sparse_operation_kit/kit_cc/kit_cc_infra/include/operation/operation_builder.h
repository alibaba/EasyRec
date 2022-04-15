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

#ifndef OPERATION_BUILDER_H
#define OPERATION_BUILDER_H

#include <memory>

#include "common.h"
#include "operation/construction_context.h"

namespace SparseOperationKit {

class Operation;
class EmbeddingLookuper;
class ParamInterface;

class Builder {
 public:
  virtual ~Builder() {}
  virtual std::shared_ptr<Operation> produce(std::shared_ptr<ConstructionContext> context) = 0;
  virtual std::shared_ptr<EmbeddingLookuper> produce(std::shared_ptr<ConstructionContext> context,
                                                     std::shared_ptr<ParamInterface> param) = 0;
};

template <typename OperationClass>
class OperationBuilder : public Builder {
 public:
  OperationBuilder() {}
  std::shared_ptr<EmbeddingLookuper> produce(std::shared_ptr<ConstructionContext> context,
                                             std::shared_ptr<ParamInterface> param) override {
    throw std::runtime_error(ErrorBase + "Not implemented.");
  }
  std::shared_ptr<Operation> produce(std::shared_ptr<ConstructionContext> context) override {
    return std::make_shared<OperationClass>(context);
  }
};

}  // namespace SparseOperationKit

#endif  // OPERATION_BUILDER_H