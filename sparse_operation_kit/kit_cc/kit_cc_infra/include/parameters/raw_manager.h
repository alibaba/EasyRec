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

#ifndef RAW_PARAMETER_MANAGER_H
#define RAW_PARAMETER_MANAGER_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "parameters/manager_interface.h"
#include "parameters/param_interface.h"
#include "parameters/raw_state.h"
#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"

namespace SparseOperationKit {

class RawManager : public ParamsManager {
 public:
  static std::shared_ptr<RawManager> Create(const std::shared_ptr<ResourcesManager>& resource_mgr);
  ~RawManager();

  void init(const size_t global_replica_id) override;
  void create_variables(const size_t local_replica_id, const std::string initializer,
                        const bool use_hashtable, const std::vector<size_t> shape,
                        const std::string name, const bool trainable,
                        const DataType dtype, const DataType key_dtype,
                        std::shared_ptr<ParamInterface>& param) override;
  void create_variables(const size_t local_replica_id, const std::shared_ptr<Tensor> initial_value,
                        const bool use_hashtable, const std::vector<size_t> shape,
                        const std::string name, const bool trainable,
                        const DataType dtype, const DataType key_dtype,
                        std::shared_ptr<ParamInterface>& param) override;
  void allocate_memory(const size_t global_replica_id) override;
  void params_initialization(const size_t global_replica_id) const override;
  void dump_to_file(const std::shared_ptr<ParamInterface>& param,
                    const std::string filepath) override;
  void restore_from_file(const std::shared_ptr<ParamInterface>& param,
                         const std::string filepath) override;
  void load_embedding_values(std::shared_ptr<ParamInterface>& param,
                             const std::shared_ptr<Tensor>& emb_values) override;
  void push_back_embedding_buffer_builder(
      const size_t local_replica_id, std::shared_ptr<EmbeddingBufferBuilder>& builder) override;

  void reserve_spaces_for_Adam() override;
  std::shared_ptr<States>& get_optimizer_states(const std::string states_name) override;
  std::shared_ptr<Tensor>& get_optimizer_state(const std::string state_name,
                                               const size_t local_replica_id) override;

  void gen_unique_name(const bool trainable, std::string& name) override;

 private:
  RawManager(const std::shared_ptr<ResourcesManager>& resource_mgr);

  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>
      buffers_;  // store trainable tensor for all gpus.
  volatile bool resized_;

  std::shared_ptr<ResourcesManager> resource_mgr_;

  std::mutex mu_;
  std::condition_variable cond_;
  std::atomic<int32_t> num_writers_waiting_{0};
  std::atomic<bool> writer_active_{false};

  std::unordered_map<std::string, std::shared_ptr<ParamInterface>>
      non_trainable_params_;  // store all embedding layers' non-trainable params
  std::unordered_map<std::string, std::shared_ptr<ParamInterface>>
      trainable_params_;  // store all embedding layers' trainable params
  std::unordered_map<std::string, std::shared_ptr<States>>
      optimizer_states_;  // store the optimizer states for all trainable params

  // the builder belong to the same device should be stored in the same vector.
  std::vector<std::vector<std::shared_ptr<EmbeddingBufferBuilder>>> embedding_buffer_builders_;
};

}  // namespace SparseOperationKit

#endif  // RAW_PARAMETER_MANAGER_H