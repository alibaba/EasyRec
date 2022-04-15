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

#ifndef HASHTABLE_NV_HASHTABLE_H
#define HASHTABLE_NV_HASHTABLE_H

#include <memory>

#include "hashtable/hashtable.h"
#include "hashtable/nv_hashtable.hpp"

namespace SparseOperationKit {

/* Adaptor for the hashtable used in HugeCTR.*/
template <typename KeyType, typename ValType>
class NvHashTable : public HashTable {
 public:
  static std::shared_ptr<NvHashTable<KeyType, ValType>> create(size_t capacity, size_t count = 0);

  virtual size_t get_and_add_value_head(size_t counter_add, cudaStream_t stream) override;
  virtual void get(const void *d_keys, void *d_vals, size_t len,
                   cudaStream_t stream) const override;
  virtual void get_insert(const void *d_keys, void *d_vals, size_t len,
                          cudaStream_t stream) override;
  virtual void insert(const void *d_keys, const void *d_vals, size_t len,
                      cudaStream_t stream) override;
  virtual size_t get_size(cudaStream_t stream) const override;
  virtual size_t get_capacity(cudaStream_t stream) const override;
  virtual size_t get_value_head(cudaStream_t stream) const override;
  virtual void dump(void *d_key, void *d_val, size_t *d_dump_counter,
                    cudaStream_t stream) const override;
  virtual bool identical_mapping() const override;

 private:
  NvHashTable(size_t capacity, size_t count = 0);

  HugeCTR::HashTable<KeyType, ValType> hashtable_;
};

}  // namespace SparseOperationKit

#endif  // HASHTABLE_NV_HASHTABLE_H