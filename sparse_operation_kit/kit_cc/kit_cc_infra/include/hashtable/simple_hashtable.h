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

#ifndef SIMPLE_HASHTABLE_H
#define SIMPLE_HASHTABLE_H

#include <memory>

#include "hashtable/hashtable.h"

namespace SparseOperationKit {

namespace HashFunctors {

struct HashFunctor {
  virtual ~HashFunctor() {}
  virtual void operator()(const void *d_key, void *d_vals, const size_t len,
                          cudaStream_t stream) = 0;
  virtual void dump(void *d_keys, void *d_vals, size_t *d_dump_counter,
                    cudaStream_t stream) const = 0;
  virtual std::unique_ptr<HashFunctor> clone(const size_t global_replica_id) = 0;
};

template <typename KeyType, typename ValType>
class Divisive : public HashFunctor {
 public:
  static std::unique_ptr<Divisive<KeyType, ValType>> create(const ValType interval,
                                                            const size_t capacity,
                                                            const size_t global_replica_id);

  void operator()(const void *d_keys, void *d_vals, const size_t len, cudaStream_t stream) override;
  void dump(void *d_keys, void *d_vals, size_t *d_dump_counter, cudaStream_t stream) const override;
  std::unique_ptr<HashFunctor> clone(const size_t global_replica_id) override;

 private:
  Divisive(const ValType interval, const size_t capacity, const size_t global_replica_id);

  const ValType interval_;
  const size_t capacity_;
  const size_t global_replica_id_;
};

}  // namespace HashFunctors

using HashFunctor_t = std::unique_ptr<HashFunctors::HashFunctor>;

/*This one is only an interface for SimpleHashtable*/
class BaseSimpleHashtable : public HashTable {
 public:
  virtual std::shared_ptr<BaseSimpleHashtable> clone(const size_t global_replica_id) = 0;
};

/*This hashtable use the HashFunctor as the hash function.*/
template <typename KeyType, typename ValType>
class SimpleHashtable : public BaseSimpleHashtable {
 public:
  static std::shared_ptr<SimpleHashtable<KeyType, ValType>> create(const size_t capacity,
                                                                   HashFunctor_t &hash_functor);

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
  virtual std::shared_ptr<BaseSimpleHashtable> clone(const size_t global_replica_id) override;

 private:
  explicit SimpleHashtable(const size_t capacity, HashFunctor_t &hash_functor);

  const size_t capacity_;
  HashFunctor_t hash_functor_;
};

}  // namespace SparseOperationKit

#endif  // SIMPLE_HASHTABLE_H