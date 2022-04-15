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

#ifndef RAW_PARAM_H
#define RAW_PARAM_H

#include <memory>
#include <vector>
#include <atomic>

#include "initializer/initializer_interface.h"
#include "parameters/param_interface.h"
#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"
#include "tensor_buffer/tensor2.hpp"

namespace SparseOperationKit {

template <typename KeyType, typename ValueType>
class RawParam : public ParamInterface {
  template <typename T>
  using Tensor2 = HugeCTR::Tensor2<T>;
  template <typename T>
  using Tensors2 = HugeCTR::Tensors2<T>;

 public:
  ~RawParam();
  static std::shared_ptr<RawParam<KeyType, ValueType>> 
    create(const std::string& initializer, const bool use_hashtable,
          const std::vector<size_t> shape,
          const std::shared_ptr<ResourcesManager>& resource_mgr,
          const std::string var_name, const bool trainable);

  // this function generates random values for initialization
  void init(const size_t global_replica_id) override;
  void set_user(std::shared_ptr<EmbeddingLayer>& embedding) override;
  std::shared_ptr<HashTable>& get_hashtable(const size_t local_replica_id) override;
  std::shared_ptr<Tensor>& get_embedding_table_tensor(const size_t local_replica_id) override;
  // this function use existing values for initialization
  void assign_initial_value(const size_t local_replica_id,
                            const std::shared_ptr<Tensor>& initial_value) override;
  void dump_to_file(const std::string filepath) override;
  void let_user_dump_to_file(const std::string filepath) override;
  void restore_from_file(const std::string filepath) override;
  void let_user_restore_from_file(const std::string filepath) override;
  void load_embedding_values(const std::shared_ptr<Tensor>& emb_values) override;
  void let_user_load_embedding_values(
      const std::shared_ptr<Tensor>& emb_values) override;
  void set_hashtable(std::shared_ptr<BaseSimpleHashtable> hashtable) override;

 private:
  RawParam(const std::string& initializer, const bool use_hashtable,
           const std::vector<size_t> shape,
           const std::shared_ptr<ResourcesManager>& resource_mgr,
           const std::string var_name, const bool trainable);

  bool is_initialized(const size_t local_replica_id) const;
  void set_initialized(const size_t local_replica_id);

  std::shared_ptr<ResourcesManager> resource_mgr_;
  std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>
      buffers_;                                         // memory buffer owned by this variable
  std::vector<std::shared_ptr<HashTable>> hashtables_;  // hashtables for all GPUs on this worker.
  Tensors2<ValueType> emb_table_tensors_;  // embedding vectors for all GPUs on this worker.
  std::vector<std::shared_ptr<Tensor>> emb_table_tensors_interface_;
  std::shared_ptr<Initializer> initializer_;
  const bool use_hashtable_;
  std::shared_ptr<EmbeddingLayer> user_;  // which embedding used this param
  std::vector<std::atomic<bool>> initialized_;         // indicates whether this variable has been initialized.
};

using RawParamCtor_t = std::function<std::shared_ptr<ParamInterface>(
      const std::string&, const bool, const std::vector<size_t>,
      const std::shared_ptr<ResourcesManager>&, const std::string, const bool)>;

RawParamCtor_t GetRawParamCtor(const DataType key_dtype, const DataType value_dtype);

}  // namespace SparseOperationKit

#endif  // RAW_PARAM_H