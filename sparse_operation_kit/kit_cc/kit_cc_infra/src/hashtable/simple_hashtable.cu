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

#include "hashtable/simple_hashtable.h"

namespace SparseOperationKit {

namespace HashFunctors {

namespace {

template <typename KeyType, typename ValType>
__global__ void divisive_kernel(const KeyType *d_keys, ValType *d_vals, const size_t len,
                                const ValType interval, const size_t capacity) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < len; i += stride) {
    assert(d_keys[i] >= 0 && "key must be greater than zero.");
    ValType val = static_cast<ValType>(d_keys[i] / interval);
    assert(val < static_cast<ValType>(capacity) && "val must be less than capacity.");
    d_vals[i] = val;
  }
}

template <typename KeyType, typename ValType>
__global__ void divisive_dump_kernel(KeyType *d_keys, ValType *d_vals, size_t *d_dump_counter,
                                     const ValType interval, const size_t capacity,
                                     const size_t global_replica_id) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < capacity; i += stride) {
    d_keys[i] = static_cast<KeyType>(i * interval + global_replica_id);
    d_vals[i] = i;
  }
  if (0 == gid) *d_dump_counter = capacity;
}

}  // anonymous namespace

template <typename KeyType, typename ValType>
Divisive<KeyType, ValType>::Divisive(const ValType interval, const size_t capacity,
                                     const size_t global_replica_id)
    : interval_(interval), capacity_(capacity), global_replica_id_(global_replica_id) {}

template <typename KeyType, typename ValType>
std::unique_ptr<Divisive<KeyType, ValType>> Divisive<KeyType, ValType>::create(
    const ValType interval, const size_t capacity, const size_t global_replica_id) {
  return std::unique_ptr<Divisive<KeyType, ValType>>(
      new Divisive<KeyType, ValType>(interval, capacity, global_replica_id));
}

template <typename KeyType, typename ValType>
void Divisive<KeyType, ValType>::operator()(const void *d_keys, void *d_vals, const size_t len,
                                            cudaStream_t stream) {
  if (0 == len) return;

  const KeyType *_d_keys = reinterpret_cast<const KeyType *>(d_keys);
  ValType *_d_vals = reinterpret_cast<ValType *>(d_vals);
  constexpr uint32_t block_size = 1024u;
  const uint32_t grid_size = (len + block_size - 1) / block_size;
  divisive_kernel<<<grid_size, block_size, 0, stream>>>(_d_keys, _d_vals, len, interval_,
                                                        capacity_);
}

template <typename KeyType, typename ValType>
void Divisive<KeyType, ValType>::dump(void *d_keys, void *d_vals, size_t *d_dump_counter,
                                      cudaStream_t stream) const {
  KeyType *_d_keys = reinterpret_cast<KeyType *>(d_keys);
  ValType *_d_vals = reinterpret_cast<ValType *>(d_vals);
  constexpr uint32_t block_size = 1024u;
  const uint32_t grid_size = (capacity_ + block_size - 1) / block_size;
  divisive_dump_kernel<<<grid_size, block_size, 0, stream>>>(
      _d_keys, _d_vals, d_dump_counter, interval_, capacity_, global_replica_id_);
}

template <typename KeyType, typename ValType>
std::unique_ptr<HashFunctor> Divisive<KeyType, ValType>::clone(const size_t global_replica_id) {
  return create(interval_, capacity_, global_replica_id);
}

template class Divisive<int64_t, size_t>;
template class Divisive<uint32_t, size_t>;

}  // namespace HashFunctors

template <typename KeyType, typename ValType>
SimpleHashtable<KeyType, ValType>::SimpleHashtable(const size_t capacity,
                                                   HashFunctor_t &hash_functor)
    : capacity_(capacity), hash_functor_(hash_functor.release()) {}

template <typename KeyType, typename ValType>
std::shared_ptr<SimpleHashtable<KeyType, ValType>> SimpleHashtable<KeyType, ValType>::create(
    const size_t capacity, HashFunctor_t &hash_functor) {
  return std::shared_ptr<SimpleHashtable<KeyType, ValType>>(
      new SimpleHashtable(capacity, hash_functor));
}

template <typename KeyType, typename ValType>
size_t SimpleHashtable<KeyType, ValType>::get_and_add_value_head(size_t counter_add,
                                                                 cudaStream_t stream) {
  // no internal counter, so that always return 0;
  return 0ul;
}

template <typename KeyType, typename ValType>
void SimpleHashtable<KeyType, ValType>::get(const void *d_keys, void *d_vals, size_t len,
                                            cudaStream_t stream) const {
  // delegate this job to hash_functor
  (*hash_functor_)(d_keys, d_vals, len, stream);
}

template <typename KeyType, typename ValType>
void SimpleHashtable<KeyType, ValType>::get_insert(const void *d_keys, void *d_vals, size_t len,
                                                   cudaStream_t stream) {
  return this->get(d_keys, d_vals, len, stream);
}

template <typename KeyType, typename ValType>
void SimpleHashtable<KeyType, ValType>::insert(const void *d_keys, const void *d_vals, size_t len,
                                               cudaStream_t stream) {
  // do nothing
  return;
}

template <typename KeyType, typename ValType>
size_t SimpleHashtable<KeyType, ValType>::get_size(cudaStream_t stream) const {
  // return its capacity
  return capacity_;
}

template <typename KeyType, typename ValType>
size_t SimpleHashtable<KeyType, ValType>::get_capacity(cudaStream_t stream) const {
  // return its capacity
  return capacity_;
}

template <typename KeyType, typename ValType>
size_t SimpleHashtable<KeyType, ValType>::get_value_head(cudaStream_t stream) const {
  return this->get_size(stream);
}

template <typename KeyType, typename ValType>
void SimpleHashtable<KeyType, ValType>::dump(void *d_key, void *d_val, size_t *d_dump_counter,
                                             cudaStream_t stream) const {
  // delegate this job to hash_functor
  hash_functor_->dump(d_key, d_val, d_dump_counter, stream);
}

template <typename KeyType, typename ValType>
bool SimpleHashtable<KeyType, ValType>::identical_mapping() const {
  return false;
}

template <typename KeyType, typename ValType>
std::shared_ptr<BaseSimpleHashtable> SimpleHashtable<KeyType, ValType>::clone(
    const size_t global_replica_id) {
  HashFunctor_t hash_func = hash_functor_->clone(global_replica_id);
  return create(capacity_, hash_func);
}

template class SimpleHashtable<int64_t, size_t>;
template class SimpleHashtable<uint32_t, size_t>;

}  // namespace SparseOperationKit