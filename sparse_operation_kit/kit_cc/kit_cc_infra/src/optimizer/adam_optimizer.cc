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

#include "optimizer/adam_optimizer.h"

#include <cmath>

#include "optimizer/update_functions.h"

namespace SparseOperationKit {

std::shared_ptr<AdamOptimizer> AdamOptimizer::create(
    optimizer_hyper_params &&hyper_params, const std::shared_ptr<ParamsManager> &params_mgr,
    std::shared_ptr<ResourcesManager> resource_mgr) {
  return std::shared_ptr<AdamOptimizer>(
      new AdamOptimizer(std::move(hyper_params), params_mgr, resource_mgr));
}

AdamOptimizer::AdamOptimizer(optimizer_hyper_params &&hyper_params,
                             const std::shared_ptr<ParamsManager> &params_mgr,
                             std::shared_ptr<ResourcesManager> resource_mgr)
    : Optimizer(params_mgr),
      resource_mgr_(resource_mgr),
      hyper_params_handler_(OptimizerHyperParamsHandler::create(std::move(hyper_params))),
      beta1_(hyper_params_handler_->get_hyper_param("beta1")),
      beta2_(hyper_params_handler_->get_hyper_param("beta2")),
      epsilon_(hyper_params_handler_->get_hyper_param("epsilon")) {}

void AdamOptimizer::create_preparers(const std::shared_ptr<EmbeddingManager> &embedding_mgr) {
  preparers_ = embedding_mgr->create_preparers_for_Adam();
}

std::shared_ptr<UpdatePreparer> &AdamOptimizer::get_update_preparer(
    const std::string variable_name) {
  auto iter = preparers_.find(variable_name);
  if (preparers_.end() == iter)
    throw std::runtime_error(ErrorBase + "Cannot find update preparer whose name is " +
                             variable_name);
  return iter->second;
}

/*This function will be called only once.*/
void AdamOptimizer::reserve_spaces() {
  // reserve states spaces, which is first-order momentum and second-order momentum
  // delegate this job to param manager
  // because the param manager can access to trainable params.
  params_mgr_->reserve_spaces_for_Adam();
}

void AdamOptimizer::apply_gradients(std::shared_ptr<ParamInterface> param,
                                    const std::shared_ptr<Tensor> grad_tensor,
                                    const std::shared_ptr<Tensor> local_indices,
                                    const size_t local_replica_id, const float learning_rate,
                                    const size_t current_step) {
  auto &emb_table_tensor = param->get_embedding_table_tensor(local_replica_id);
  const auto variable_name = param->get_var_name();
  const auto &stream = resource_mgr_->get_local_gpu(local_replica_id)->get_stream();

  // get its states
  auto &m_states = params_mgr_->get_optimizer_state(variable_name + "_1order", local_replica_id);
  auto &v_states = params_mgr_->get_optimizer_state(variable_name + "_2order", local_replica_id);

  // get the update preparer for this variable, and do preparation
  auto &preparer = get_update_preparer(variable_name);
  (*preparer)(local_indices, local_replica_id);
  auto &indices_unique_indexes = preparer->get_indices_unique_indexes(local_replica_id);
  auto &sorted_position = preparer->get_sorted_position(local_replica_id);
  auto &sorted_indices = preparer->get_sorted_indices(local_replica_id);
  const uint32_t unique_indices_num = preparer->get_unique_indices_num(local_replica_id);

  // because TF will update step after the applying is done.
  const size_t next_step = current_step + 1l;

  const float alpha_t = learning_rate * std::sqrt(1 - std::pow(beta2_, next_step)) /
                        (1 - std::pow(beta1_, next_step));

  // update first-order momentum and second-order momentum
  opt_adam_update_states_global(
      indices_unique_indexes->GetPtrWithType<size_t>(), sorted_position->GetPtrWithType<int64_t>(),
      sorted_indices->GetPtrWithType<int64_t>(), unique_indices_num,
      grad_tensor->GetPtrWithType<float>(), param->get_embedding_vec_size(), beta1_, beta2_,
      m_states->GetPtrWithType<float>(), v_states->GetPtrWithType<float>(), stream);

  // update embedding variable
  opt_adam_update_param_global(
      param->get_embedding_vec_size(), param->get_max_vocabulary_size_per_gpu(), alpha_t, beta1_,
      beta2_, epsilon_, m_states->GetPtrWithType<float>(), v_states->GetPtrWithType<float>(),
      emb_table_tensor->GetPtrWithType<float>(), stream);
}

}  // namespace SparseOperationKit