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

#include "parameters/raw_manager.h"

#include <thread>

#include "parameters/raw_param.h"
#include "parameters/raw_state.h"

namespace SparseOperationKit {

RawManager::RawManager(const std::shared_ptr<ResourcesManager>& resource_mgr)
    : resized_(false), resource_mgr_(resource_mgr), mu_(), cond_() {}

RawManager::~RawManager() {}

std::shared_ptr<RawManager> RawManager::Create(
    const std::shared_ptr<ResourcesManager>& resource_mgr) {
  return std::shared_ptr<RawManager>(new RawManager(resource_mgr));
}

void RawManager::init(const size_t global_replica_id) {
  // resize buffers_ vector
  auto helper = [this]() {
    if (resized_) throw std::runtime_error(ErrorBase + "RawManager had been initialized.");
    buffers_.resize(resource_mgr_->get_local_gpu_count());
    embedding_buffer_builders_.resize(resource_mgr_->get_local_gpu_count());
    resized_ = true;
  };

  resource_mgr_->blocking_call_once(helper);

  // new GeneralBuffer object for local GPU by its own.
  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
  buffers_[local_replica_id] = HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>::create();
}

void RawManager::gen_unique_name(const bool trainable, std::string& name) {
  // generate a new name when there already exists an variable whose name is "name"
  if (trainable) {
    while (trainable_params_.find(name) != trainable_params_.end()) {
      auto name_vec = str_split(name, "_");
      int32_t num = string2num(*(name_vec.rbegin()));
      if (-1 == num) {
        name = name + "_" + std::to_string(1);
      } else {
        *(name_vec.rbegin()) = std::to_string(num + 1);
        name = strs_concat(name_vec, "_");
      }
    }
  } else {
    while (non_trainable_params_.find(name) != non_trainable_params_.end()) {
      auto name_vec = str_split(name, "_");
      int32_t num = string2num(*(name_vec.rbegin()));
      if (-1 == num) {
        name = name + "_" + std::to_string(1);
      } else {
        *(name_vec.rbegin()) = std::to_string(num + 1);
        name = strs_concat(name_vec, "_");
      }
    }
  }
}

void RawManager::create_variables(const size_t local_replica_id, const std::string initializer,
                                  const bool use_hashtable, const std::vector<size_t> shape,
                                  const std::string name, const bool trainable,
                                  const DataType dtype, const DataType key_dtype,
                                  std::shared_ptr<ParamInterface>& param) {
  // If shape is the same as the previous one,
  // then the previous param should be returned,
  // rather than creating a new one.
  static std::vector<size_t> previous_shape;
  static std::shared_ptr<ParamInterface> previous_param;
  if (shape == previous_shape) {
    // return the previous param
    param = previous_param;
  } else {
    std::shared_ptr<ParamInterface> raw_param{nullptr};
    {  // begin write
      num_writers_waiting_.fetch_add(1, std::memory_order_acq_rel);
      std::unique_lock<std::mutex> lock(mu_);
      // wait until no active writer
      cond_.wait(lock, [this] { return !writer_active_.load(std::memory_order_acquire); });
      num_writers_waiting_.fetch_sub(1, std::memory_order_acq_rel);
      writer_active_.store(true, std::memory_order_release);

      // create variable
      // variable will have its own memory buffer
      raw_param = ParamInterface::CreateParam(/*param_type=*/ParamType::RawParam,
                                              /*initializer=*/initializer,
                                              /*use_hashtable=*/use_hashtable,
                                              /*shape=*/shape,
                                              /*resource_mgr=*/resource_mgr_,
                                              /*var_name=*/name,
                                              /*trainable=*/trainable,
                                              /*key_dtype=*/key_dtype,
                                              /*value_dtype=*/dtype);
      if (trainable) {
        trainable_params_.emplace(std::make_pair(name, raw_param));
      } else {
        non_trainable_params_.emplace(std::make_pair(name, raw_param));
      }

      // end write
      writer_active_.store(false, std::memory_order_release);
      cond_.notify_one();
    }

    // update previous state
    previous_param = raw_param;
    previous_shape = shape;

    // set output
    param = raw_param;
  }  // if shape == previous_shape

  MESSAGE("Created embedding variable whose name is " + name);
}

void RawManager::create_variables(const size_t local_replica_id,
                                  const std::shared_ptr<Tensor> initial_value,
                                  const bool use_hashtable, const std::vector<size_t> shape,
                                  const std::string name, const bool trainable,
                                  const DataType dtype, const DataType key_dtype,
                                  std::shared_ptr<ParamInterface>& param) {
  // create variable
  create_variables(local_replica_id, /*initializer=*/"ones", use_hashtable, shape, 
                   name, trainable, dtype, key_dtype, param);
  // set initial_value
  param->assign_initial_value(local_replica_id, initial_value);
}

void RawManager::allocate_memory(const size_t global_replica_id) {
  const size_t local_replica_id = global_replica_id % resource_mgr_->get_local_gpu_count();
  if (local_replica_id >= buffers_.size())
    throw std::runtime_error(ErrorBase +
                             "The local_replica_id is out of the range of buffers.size().");

  if (buffers_[local_replica_id]->allocated()) return;  // memory has already been allocated.

  {  // begin read
    std::unique_lock<std::mutex> lock(mu_);
    // wait until no waiting writers and no active writers
    cond_.wait(lock, [this] {
      return (num_writers_waiting_.load(std::memory_order_acquire) == 0 &&
              !writer_active_.load(std::memory_order_acquire));
    });
  }

  buffers_[local_replica_id]->allocate();

  // do parameters initialization
  params_initialization(global_replica_id);

  // update EmbeddingTensorBuffer which is used by TF ops.
  for (auto builder : embedding_buffer_builders_[local_replica_id]) {
    builder->build_buffer();
  }
}

void RawManager::params_initialization(const size_t global_replica_id) const {
  // initialize trainable parameters
  for (auto iter : trainable_params_) {
    iter.second->init(global_replica_id);
  }
  // TODO: whether should initialize non_trainable_params ???

  // initialize optimizer states
  for (auto iter : optimizer_states_) {
    iter.second->init(global_replica_id);
  }
}

void RawManager::dump_to_file(const std::shared_ptr<ParamInterface>& param,
                              const std::string filepath) {
  try {
    MESSAGE("Saving " + param->get_var_name() + " to " + filepath + "..");
    resource_mgr_->sync_all_workers();
    // step 1: save param to file.
    param->dump_to_file(filepath);

    // step 2: save operations' content from param's user to file
    param->let_user_dump_to_file(filepath);
    resource_mgr_->sync_all_workers();
    MESSAGE("Saved " + param->get_var_name() + " to " + filepath + ".");
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
}

void RawManager::restore_from_file(const std::shared_ptr<ParamInterface>& param,
                                   const std::string filepath) {
  try {
    MESSAGE("Restoring " + param->get_var_name() + " from " + filepath + "..");
    resource_mgr_->sync_all_workers();
    // step 1: restore param from file
    param->restore_from_file(filepath);

    // step 2: restore operations' content from file
    param->let_user_restore_from_file(filepath);
    resource_mgr_->sync_all_workers();
    MESSAGE("Restored " + param->get_var_name() + " from " + filepath + ".");
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
}

void RawManager::load_embedding_values(std::shared_ptr<ParamInterface>& param,
                                       const std::shared_ptr<Tensor>& emb_values) {
  MESSAGE("Loading embedding values to Variable: " + param->get_var_name() + "...");
  resource_mgr_->sync_all_workers();

  // step 1: let param to modify its memory states with these tensor_list
  param->load_embedding_values(emb_values);

  // step 2: let operations modify their memory states with these tensor_list
  param->let_user_load_embedding_values(emb_values);

  resource_mgr_->sync_all_workers();
  MESSAGE("Loaded embedding values to Variable: " + param->get_var_name() + ".");
}

void RawManager::push_back_embedding_buffer_builder(
    const size_t local_replica_id, std::shared_ptr<EmbeddingBufferBuilder>& builder) {
  std::lock_guard<std::mutex> lock(mu_);
  embedding_buffer_builders_[local_replica_id].emplace_back(builder);
}

// This function should be called once.
void RawManager::reserve_spaces_for_Adam() {
  // FIXME: should use template arg for RawStates??
  // iterate all trainable params
  for (const auto& params : trainable_params_) {
    auto param_name = params.first;
    auto param_interface = params.second;
    const std::vector<size_t> param_shape{param_interface->get_max_vocabulary_size_per_gpu(),
                                          param_interface->get_embedding_vec_size()};
    {  // first-order momentum for Adam
      const std::string m_name = param_name + "_1order";
      auto m_states = RawStates::create(param_shape, resource_mgr_, buffers_);
      optimizer_states_.emplace(std::make_pair(m_name, m_states));
    }

    {  // second-order momentum for Adam
      const std::string v_name = param_name + "_2order";
      auto v_states = RawStates::create(param_shape, resource_mgr_, buffers_);
      optimizer_states_.emplace(std::make_pair(v_name, v_states));
    }
  }  // for param in trainable params
}

std::shared_ptr<States>& RawManager::get_optimizer_states(const std::string states_name) {
  const auto iter = optimizer_states_.find(states_name);
  if (optimizer_states_.end() == iter)
    throw std::runtime_error(ErrorBase +
                             "Cannot find optimizer states whose name is: " + states_name);
  return iter->second;
}

std::shared_ptr<Tensor>& RawManager::get_optimizer_state(const std::string state_name,
                                                         const size_t local_replica_id) {
  auto states = get_optimizer_states(state_name);
  return states->get_tensor(local_replica_id);
}

}  // namespace SparseOperationKit