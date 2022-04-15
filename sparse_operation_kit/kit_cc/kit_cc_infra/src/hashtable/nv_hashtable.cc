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

#include "hashtable/nv_hashtable.h"

/*Implementations of adaptor for the hashtable used in HugeCTR.*/
namespace SparseOperationKit {

template <typename KeyType, typename ValType>
NvHashTable<KeyType, ValType>::NvHashTable(size_t capacity, size_t count)
    : hashtable_(capacity, count) {}

template <typename KeyType, typename ValType>
std::shared_ptr<NvHashTable<KeyType, ValType>> NvHashTable<KeyType, ValType>::create(
    size_t capacity, size_t count) {
  return std::shared_ptr<NvHashTable<KeyType, ValType>>(
      new NvHashTable<KeyType, ValType>(capacity, count));
}

template <typename KeyType, typename ValType>
size_t NvHashTable<KeyType, ValType>::get_and_add_value_head(size_t counter_add,
                                                             cudaStream_t stream) {
  return hashtable_.get_and_add_value_head(counter_add, stream);
}

template <typename KeyType, typename ValType>
void NvHashTable<KeyType, ValType>::get(const void *d_keys, void *d_vals, size_t len,
                                        cudaStream_t stream) const {
  const KeyType *_d_keys = reinterpret_cast<const KeyType *>(d_keys);
  ValType *_d_vals = reinterpret_cast<ValType *>(d_vals);
  return hashtable_.get(_d_keys, _d_vals, len, stream);
}

template <typename KeyType, typename ValType>
void NvHashTable<KeyType, ValType>::get_insert(const void *d_keys, void *d_vals, size_t len,
                                               cudaStream_t stream) {
  const KeyType *_d_keys = reinterpret_cast<const KeyType *>(d_keys);
  ValType *_d_vals = reinterpret_cast<ValType *>(d_vals);
  return hashtable_.get_insert(_d_keys, _d_vals, len, stream);
}

template <typename KeyType, typename ValType>
void NvHashTable<KeyType, ValType>::insert(const void *d_keys, const void *d_vals, size_t len,
                                           cudaStream_t stream) {
  const KeyType *_d_keys = reinterpret_cast<const KeyType *>(d_keys);
  const ValType *_d_vals = reinterpret_cast<const ValType *>(d_vals);
  return hashtable_.insert(_d_keys, _d_vals, len, stream);
}

template <typename KeyType, typename ValType>
size_t NvHashTable<KeyType, ValType>::get_size(cudaStream_t stream) const {
  return hashtable_.get_size(stream);
}

template <typename KeyType, typename ValType>
size_t NvHashTable<KeyType, ValType>::get_capacity(cudaStream_t stream) const {
  return hashtable_.get_capacity();
}

template <typename KeyType, typename ValType>
size_t NvHashTable<KeyType, ValType>::get_value_head(cudaStream_t stream) const {
  return hashtable_.get_value_head(stream);
}

template <typename KeyType, typename ValType>
void NvHashTable<KeyType, ValType>::dump(void *d_key, void *d_val, size_t *d_dump_counter,
                                         cudaStream_t stream) const {
  KeyType *_d_key = reinterpret_cast<KeyType *>(d_key);
  ValType *_d_val = reinterpret_cast<ValType *>(d_val);
  return hashtable_.dump(_d_key, _d_val, d_dump_counter, stream);
}

template <typename KeyType, typename ValType>
bool NvHashTable<KeyType, ValType>::identical_mapping() const {
  return false;
}

template class NvHashTable<int64_t, size_t>;
template class NvHashTable<uint32_t, size_t>;

}  // namespace SparseOperationKit