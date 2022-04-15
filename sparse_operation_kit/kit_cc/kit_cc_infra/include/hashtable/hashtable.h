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

#ifndef HASHTABLE_HASHTABLE_H
#define HASHTABLE_HASHTABLE_H

#include "cuda_runtime_api.h"

namespace SparseOperationKit {

class HashTable {
 public:
  virtual ~HashTable() = default;

  /**
   * Add a number to the head of the value. This will add the given value to the
   * current value of the device counter.
   * @param counter_add the new counter value to be added.
   */
  virtual size_t get_and_add_value_head(size_t counter_add, cudaStream_t stream) = 0;

  /**
   * The get function for hash table. "get" means fetching some values indexed
   * by the given keys.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be got from the hash table.
   * @param stream the cuda stream for this operation.
   */
  virtual void get(const void *d_keys, void *d_vals, size_t len, cudaStream_t stream) const = 0;

  /**
   * The get_insert function for hash table. "get_insert" means if we can find
   * the keys in the hash table, the values indexed by the keys will be returned,
   * which is known as a "get" operation; Otherwise, the not-found keys together
   * with the values computed by the current device counter automatically will be
   * inserted into the hash table.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be got or inserted into the hash table.
   * @param stream the cuda stream for this operation.
   */
  virtual void get_insert(const void *d_keys, void *d_vals, size_t len, cudaStream_t stream) = 0;

  /**
   * The insert function for hash table. "insert" means putting some new <key,value> pairs
   * into the current hash table.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be inserted into the hash table.
   * @param stream the cuda stream for this operation.
   */
  virtual void insert(const void *d_keys, const void *d_vals, size_t len, cudaStream_t stream) = 0;

  /**
   * Get the current size of the hash table. Size is also known as the number
   * of <key,value> pairs.
   * @param stream the cuda stream for this operation.
   */
  virtual size_t get_size(cudaStream_t stream) const = 0;

  /**
   * Get the capacity of the hash table. Size is also known as the number
   * of <key,value> pairs.
   * @param stream the cuda stream for this operation.
   */
  virtual size_t get_capacity(cudaStream_t stream) const = 0;

  /**
   * Get the head of the value from the device counter. It's equal to the
   * number of the <key,value> pairs in the hash table.
   */
  virtual size_t get_value_head(cudaStream_t stream) const = 0;

  /**
   * The dump function for hash table. "dump" means getting some of the <key,value>
   * pairs from the hash table and copying them to the corresponding memory buffer.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param d_dump_counter a temp device pointer to store the dump counter.
   * @param stream the cuda stream for this operation.
   */
  virtual void dump(void *d_key, void *d_val, size_t *d_dump_counter,
                    cudaStream_t stream) const = 0;

  /**
   * decide whether this hashtable use identical mapping
   */
  virtual bool identical_mapping() const = 0;
};

}  // namespace SparseOperationKit

#endif  // HASHTABLE_HASHTABLE_H