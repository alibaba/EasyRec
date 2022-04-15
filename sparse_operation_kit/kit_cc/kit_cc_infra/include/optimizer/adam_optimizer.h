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

#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "optimizer/optimizer_interface.h"
#include "optimizer/update_preparer.h"
#include "resources/manager.h"

namespace SparseOperationKit {

class AdamOptimizer : public Optimizer {
 public:
  static std::shared_ptr<AdamOptimizer> create(optimizer_hyper_params&& hyper_params,
                                               const std::shared_ptr<ParamsManager>& params_mgr,
                                               std::shared_ptr<ResourcesManager> resource_mgr);

  void apply_gradients(std::shared_ptr<ParamInterface> param,
                       const std::shared_ptr<Tensor> grad_tensor,
                       const std::shared_ptr<Tensor> local_indices, const size_t local_replica_id,
                       const float learning_rate, const size_t current_step) override;

  void create_preparers(const std::shared_ptr<EmbeddingManager>& embedding_mgr) override;
  void reserve_spaces() override;

 private:
  AdamOptimizer(optimizer_hyper_params&& hyper_params,
                const std::shared_ptr<ParamsManager>& params_mgr,
                std::shared_ptr<ResourcesManager> resource_mgr);

  std::shared_ptr<UpdatePreparer>& get_update_preparer(const std::string variable_name);

  std::shared_ptr<ResourcesManager> resource_mgr_;

  std::unordered_map<std::string, std::shared_ptr<UpdatePreparer>> preparers_;

  OptimizerHyperParamsHandler_t hyper_params_handler_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

}  // namespace SparseOperationKit

#endif  // ADAM_OPTIMIZER_H