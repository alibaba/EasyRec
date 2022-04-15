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

#ifndef HASHTABLE_IDENTITY_HASHTABLE_H
#define HASHTABLE_IDENTITY_HASHTABLE_H

#include <memory>

#include "hashtable/hashtable.h"

namespace SparseOperationKit {

template <typename KeyType, typename ValType>
class IdentityHashTable : public HashTable {
 public:
  static std::shared_ptr<IdentityHashTable<KeyType, ValType>> create(const size_t capacity);

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
  explicit IdentityHashTable(const size_t capacity);

  const size_t capacity_;
};

}  // namespace SparseOperationKit

#endif  // HASHTABLE_IDENTITY_HASHTABLE_H