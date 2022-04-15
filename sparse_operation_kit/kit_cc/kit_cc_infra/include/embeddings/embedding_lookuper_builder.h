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

#ifndef EMBEDDING_LOOKUPER_BUILDER_H
#define EMBEDDING_LOOKUPER_BUILDER_H

#include "embeddings/embedding_lookuper.h"
#include "operation/operation_builder.h"

namespace SparseOperationKit {

template <typename Lookuper>
class EmbeddingLookuperBuilder : public Builder {
 public:
  EmbeddingLookuperBuilder() {}
  std::shared_ptr<EmbeddingLookuper> produce(ConstructionContext_t context,
                                             std::shared_ptr<ParamInterface> param) override {
    return std::make_shared<Lookuper>(context, param);
  }

  std::shared_ptr<Operation> produce(std::shared_ptr<ConstructionContext> context) override {
    throw std::runtime_error(ErrorBase + "Not implemented.");
  }
};

}  // namespace SparseOperationKit

#endif  // EMBEDDING_LOOKUPER_BUILDER_H