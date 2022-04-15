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

#ifndef EMBEDDING_LOOKUPER_H
#define EMBEDDING_LOOKUPER_H

#include "operation/operation.h"
#include "parameters/param_interface.h"

namespace SparseOperationKit {

class EmbeddingLookuper : public Operation {
 public:
  EmbeddingLookuper(ConstructionContext_t construction_context,
                    std::shared_ptr<ParamInterface> param);

  /** help param to save parameters from GPU to file
  @param keys, CPU tensor stored keys from all utilized GPUs.
  @param embedding_values, CPU tensor stored embedding_values from all utilized GPUs.
      And there is a mapping between key and embedding_value.
  @param num_total_keys, how many items in keys and embedding_values.
  */
  virtual void save_params(std::shared_ptr<Tensor> &keys, 
                           std::shared_ptr<Tensor> &embedding_values,
                           size_t &num_total_keys) const = 0;

  /** help Param to restore parameters from CPU tensor to GPU memory
  @param keys, CPU tensor stored keys for all GPUs.
  @param embedding_values, CPU tensor stored embedding_values for all GPUs.
      And there is a mapping between key and embedding_value.
  @param num_total_keys, how many items in keys and embedding_values.
  */
  virtual void restore_params(const std::shared_ptr<Tensor> &keys,
                              const std::shared_ptr<Tensor> &embedding_values,
                              const size_t num_total_keys) = 0;

 protected:
  std::shared_ptr<ParamInterface> param_;
};

}  // namespace SparseOperationKit

#endif  // EMBEDDING_LOOKUPER_H