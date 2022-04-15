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

#ifndef PARAM_INTERFACE_H
#define PARAM_INTERFACE_H

#include "hashtable/hashtable.h"
#include "hashtable/simple_hashtable.h"
#include "parameters/state_interface.h"
#include "common.h"

namespace SparseOperationKit {

class EmbeddingLayer;
class ResourcesManager;
/*
 * This class represents the variables shared to multiple GPUs.
 */
class ParamInterface : public States {
 public:
  // helper function to create Param Implementation
  static std::shared_ptr<ParamInterface> CreateParam(const ParamType param_type,
                                                     const std::string& initializer,
                                                     const bool use_hashtable,
                                                     const std::vector<size_t> shape,
                                                     const std::shared_ptr<ResourcesManager>& resource_mgr,
                                                     const std::string var_name, 
                                                     const bool trainable,
                                                     const DataType key_dtype, 
                                                     const DataType value_dtype);

  virtual ~ParamInterface() {}
  ParamInterface(const size_t max_vocabulary_size_per_gpu, const size_t embedding_vec_size,
                 const bool trainable, const std::string var_name,
                 const DataType dtype, const DataType key_dtype);
  virtual std::shared_ptr<HashTable>& get_hashtable(const size_t local_replica_id) = 0;
  virtual std::shared_ptr<Tensor>& get_embedding_table_tensor(const size_t local_replica_id) = 0;
  std::shared_ptr<Tensor>& get_tensor(const size_t local_replica_id) override;
  virtual void assign_initial_value(const size_t local_replica_id,
                                    const std::shared_ptr<Tensor>& initial_value) = 0;
  virtual void dump_to_file(const std::string filepath) = 0;
  virtual void restore_from_file(const std::string filepath) = 0;
  virtual void load_embedding_values(const std::shared_ptr<Tensor>& emb_values) = 0;
  // It is not compulsory for the subclass to override this function.
  virtual size_t get_max_vocabulary_size_per_gpu() const;
  virtual size_t get_embedding_vec_size() const;
  virtual bool trainable() const;
  virtual std::string get_var_name() const;
  virtual DataType dtype() const;
  virtual DataType key_dtype() const;

  virtual void set_user(std::shared_ptr<EmbeddingLayer>& embedding);
  virtual void let_user_dump_to_file(const std::string filepath);
  virtual void let_user_restore_from_file(const std::string filepath);
  virtual void let_user_load_embedding_values(
      const std::shared_ptr<Tensor>& emb_values);
  virtual void set_hashtable(std::shared_ptr<BaseSimpleHashtable> hashtable);

 private:
  const size_t max_vocabulary_size_per_gpu_;
  const size_t embedding_vec_size_;
  const bool trainable_;
  const std::string var_name_;
  DataType dtype_;
  DataType key_dtype_;
};

struct TypeIdentity {
  TypeIdentity(const DataType key_dtype, const DataType value_dtype);
  const DataType key_dtype_;
  const DataType value_dtype_;
};

struct TypeIdentityHash {
  size_t operator()(const TypeIdentity& type_id) const;
};

struct TypeIdentityEqual {
  bool operator()(const TypeIdentity& lid, const TypeIdentity& rid) const;
};

}  // namespace SparseOperationKit

#endif  // PARAM_INTERFACE_H