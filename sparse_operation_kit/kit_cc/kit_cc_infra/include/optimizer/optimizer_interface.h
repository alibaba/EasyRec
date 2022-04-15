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

#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include <memory>

#include "common.h"
#include "embeddings/manager.h"
#include "parameters/manager_interface.h"

namespace SparseOperationKit {

class Optimizer {
 public:
  static std::shared_ptr<Optimizer> Get(const OptimizerType optimizer_type,
                                        optimizer_hyper_params&& hyper_params,
                                        const std::shared_ptr<ParamsManager>& params_mgr,
                                        std::shared_ptr<ResourcesManager>& resource_mgr);

  explicit Optimizer(const std::shared_ptr<ParamsManager>& params_mgr);
  Optimizer() = delete;

  virtual ~Optimizer() {}

  /*delegate this job to embedding manager, because it can
   * access to some embedding attributes
   */
  virtual void create_preparers(const std::shared_ptr<EmbeddingManager>& embedding_mgr) = 0;

  /*delegate this job to params manager, this function must
   * call the right API of params manager to reserve_spaces
   */
  virtual void reserve_spaces() = 0;

  virtual void apply_gradients(std::shared_ptr<ParamInterface> param,
                               const std::shared_ptr<Tensor> grad_tensor,
                               const std::shared_ptr<Tensor> local_indices,
                               const size_t local_replica_id, const float learning_rate,
                               const size_t current_step) = 0;

 protected:
  std::shared_ptr<ParamsManager> params_mgr_;
};

void StoreOptimizerInVariantTensor(const std::shared_ptr<Optimizer>& optimizer,
                                   tensorflow::Tensor* tensor);

}  // namespace SparseOperationKit

#endif  // OPTIMIZER_INTERFACE_H