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

#ifndef PARAMETER_MANAGER_INTERFACE_H
#define PARAMETER_MANAGER_INTERFACE_H

#include <string>
#include <vector>

#include "parameters/param_interface.h"
#include "parameters/state_interface.h"
#include "tensor_buffer/embedding_buffer.h"

namespace SparseOperationKit {

class ParamsManager {
 public:
  virtual ~ParamsManager() {}
  virtual void init(const size_t global_replica_id) = 0;
  virtual void create_variables(const size_t local_replica_id, const std::string initializer,
                                const bool use_hashtable, const std::vector<size_t> shape,
                                const std::string name, const bool trainable,
                                const DataType dtype, const DataType key_dtype,
                                std::shared_ptr<ParamInterface>& param) = 0;
  virtual void create_variables(const size_t local_replica_id,
                                const std::shared_ptr<Tensor> initial_value,
                                const bool use_hashtable, const std::vector<size_t> shape,
                                const std::string name, const bool trainable,
                                const DataType dtype, const DataType key_dtype,
                                std::shared_ptr<ParamInterface>& param) = 0;
  virtual void allocate_memory(const size_t global_replica_id) = 0;
  virtual void params_initialization(const size_t global_replica_id) const = 0;
  virtual void dump_to_file(const std::shared_ptr<ParamInterface>& param,
                            const std::string filepath) = 0;
  virtual void restore_from_file(const std::shared_ptr<ParamInterface>& param,
                                 const std::string filepath) = 0;
  virtual void load_embedding_values(std::shared_ptr<ParamInterface>& param,
                                     const std::shared_ptr<Tensor>& emb_values) = 0;
  virtual void push_back_embedding_buffer_builder(const size_t local_replica_id,
                                                  std::shared_ptr<EmbeddingBufferBuilder>& builder);

  virtual void reserve_spaces_for_Adam() = 0;
  virtual std::shared_ptr<States>& get_optimizer_states(const std::string states_name) = 0;
  virtual std::shared_ptr<Tensor>& get_optimizer_state(const std::string state_name,
                                                       const size_t local_replica_id) = 0;

  virtual void gen_unique_name(const bool trainable, std::string& name) = 0;
};

}  // namespace SparseOperationKit

#endif  // PARAMETER_MANAGER_INTERFACE_H