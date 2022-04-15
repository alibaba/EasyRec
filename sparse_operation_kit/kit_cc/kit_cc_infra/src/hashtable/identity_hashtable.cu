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

#include "common.cuh"
#include "hashtable/identity_hashtable.h"

namespace SparseOperationKit {

template <typename KeyType, typename ValType>
IdentityHashTable<KeyType, ValType>::IdentityHashTable(const size_t capacity)
    : capacity_(capacity) {}

template <typename KeyType, typename ValType>
std::shared_ptr<IdentityHashTable<KeyType, ValType>> IdentityHashTable<KeyType, ValType>::create(
    const size_t capacity) {
  return std::shared_ptr<IdentityHashTable<KeyType, ValType>>(new IdentityHashTable(capacity));
}

template <typename KeyType, typename ValType>
size_t IdentityHashTable<KeyType, ValType>::get_and_add_value_head(size_t counter_add,
                                                                   cudaStream_t stream) {
  // No internal counter, so that always return 0.
  return 0ul;
}

template <typename KeyType, typename ValType>
void IdentityHashTable<KeyType, ValType>::get(const void *d_keys, void *d_vals, size_t len,
                                              cudaStream_t stream) const {
  if (0 == len) return;
  // only translate the type of key's to that of value's.
  const KeyType *_d_keys = reinterpret_cast<const KeyType *>(d_keys);
  ValType *_d_vals = reinterpret_cast<ValType *>(d_vals);

  const size_t grid_size = (len - 1) / 256ul + 1;
  auto type_conversion = [] __device__(KeyType value) { return static_cast<ValType>(value); };
  transform_array<<<grid_size, 256, 0, stream>>>(_d_keys, _d_vals, len, type_conversion);
}

template <typename KeyType, typename ValType>
void IdentityHashTable<KeyType, ValType>::get_insert(const void *d_keys, void *d_vals, size_t len,
                                                     cudaStream_t stream) {
  return this->get(d_keys, d_vals, len, stream);
}

template <typename KeyType, typename ValType>
void IdentityHashTable<KeyType, ValType>::insert(const void *d_keys, const void *d_vals, size_t len,
                                                 cudaStream_t stream) {
  // do nothing.
  return;
}

template <typename KeyType, typename ValType>
size_t IdentityHashTable<KeyType, ValType>::get_size(cudaStream_t stream) const {
  // return its capacity
  return capacity_;
}

template <typename KeyType, typename ValType>
size_t IdentityHashTable<KeyType, ValType>::get_capacity(cudaStream_t stream) const {
  return capacity_;
}

template <typename KeyType, typename ValType>
size_t IdentityHashTable<KeyType, ValType>::get_value_head(cudaStream_t stream) const {
  return this->get_size(stream);
}

template <typename KeyType, typename ValType>
void IdentityHashTable<KeyType, ValType>::dump(void *d_key, void *d_val, size_t *d_dump_counter,
                                               cudaStream_t stream) const {
  throw std::runtime_error(ErrorBase +
                           "Not Implemented. "
                           "Cause I don't record keys during get / get_insert and "
                           "don't know which embedding lookuper used me, "
                           "therefore I don't know how the keys are distributed among GPUs.");
}

template <typename KeyType, typename ValType>
bool IdentityHashTable<KeyType, ValType>::identical_mapping() const {
  return true;
}

template class IdentityHashTable<int64_t, size_t>;
template class IdentityHashTable<uint32_t, size_t>;

}  // namespace SparseOperationKit
