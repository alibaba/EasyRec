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

#include "hashtable/cudf/concurrent_unordered_map.cuh"
#include "hashtable/nv_hashtable.hpp"
#include "thrust/pair.h"

namespace HugeCTR {

namespace {

template <typename value_type>
struct ReplaceOp {
  // constexpr static value_type IDENTITY{0};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_kernel(Table* table, const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals, size_t len) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void search_kernel(Table* table, const typename Table::key_type* const keys,
                              typename Table::mapped_type* const vals, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    assert(it != table->end() && "error: can't find key");
    vals[i] = it->second;
  }
}

template <typename Table>
__global__ void get_insert_kernel(Table* table, const typename Table::key_type* const keys,
                                  typename Table::mapped_type* const vals, size_t len,
                                  size_t* d_counter) {
  ReplaceOp<typename Table::mapped_type> op;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->get_insert(keys[i], op, d_counter);
    assert(it != table->end() && "error: get_insert fails: table is full");
    vals[i] = it->second;
  }
}

template <typename Table, typename KeyType>
__global__ void size_kernel(const Table* table, const size_t hash_capacity, size_t* container_size,
                            KeyType unused_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  /* Whether the bucket mapping to the current thread is empty? do nothing : Atomically add to
   * counter */
  if (i < hash_capacity) {
    typename Table::value_type val = load_pair_vectorized(table->data() + i);
    if (val.first != unused_key) {
      atomicAdd(&block_acc, 1);
    }
  }
  __syncthreads();

  /* Atomically reduce block counter to global conuter */
  if (threadIdx.x == 0) {
    atomicAdd(container_size, block_acc);
  }
}

template <typename KeyType, typename ValType, typename Table>
__global__ void dump_kernel(KeyType* d_key, ValType* d_val, const Table* table, const size_t offset,
                            const size_t search_length, size_t* d_dump_counter,
                            KeyType unused_key) {
  // inter-block gathered key, value and counter. Global conuter for storing shared memory into
  // global memory.
  //__shared__ KeyType block_result_key[BLOCK_SIZE_];
  //__shared__ ValType block_result_val[BLOCK_SIZE_];
  extern __shared__ unsigned char s[];
  KeyType* smem = (KeyType*)s;
  KeyType* block_result_key = smem;
  ValType* block_result_val = (ValType*)&(smem[blockDim.x]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  // Each thread gather the key and value from bucket assigned to them and store them into shared
  // mem.
  if (i < search_length) {
    typename Table::value_type val = load_pair_vectorized(table->data() + offset + i);
    if (val.first != unused_key) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = val.first;
      block_result_val[local_index] = val.second;
    }
  }
  __syncthreads();

  // Each block request a unique place in global memory buffer, this is the place where shared
  // memory store back to.
  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  // Each thread store one bucket's data back to global memory, d_dump_counter is how many buckets
  // in total dumped.
  if (threadIdx.x < block_acc) {
    d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
    d_val[global_acc + threadIdx.x] = block_result_val[threadIdx.x];
  }
}

}  // namespace

template <typename KeyType, typename ValType>
class HashTableContainer
    : public concurrent_unordered_map<KeyType, ValType, std::numeric_limits<KeyType>::max()> {
 public:
  HashTableContainer(size_t capacity)
      : concurrent_unordered_map<KeyType, ValType, std::numeric_limits<KeyType>::max()>(
            capacity, std::numeric_limits<ValType>::max()) {}
};

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity, size_t count) : capacity_(capacity) {
  container_ =
      new HashTableContainer<KeyType, ValType>(static_cast<size_t>(capacity / LOAD_FACTOR));

  // Allocate device-side counter and copy user input to it
  CK_CUDA(cudaMalloc((void**)&d_counter_, sizeof(size_t)));
  CK_CUDA(cudaMalloc((void**)&d_container_size_, sizeof(size_t)));
  CK_CUDA(cudaMemcpy(d_counter_, &count, sizeof(size_t), cudaMemcpyHostToDevice));
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  try {
    delete container_;
    // De-allocate device-side counter
    CK_CUDA(cudaFree(d_counter_));
    CK_CUDA(cudaFree(d_container_size_));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys, const ValType* d_vals, size_t len,
                                         cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get_insert(const KeyType* d_keys, ValType* d_vals, size_t len,
                                             cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  get_insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len,
                                                           d_counter_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, ValType* d_vals, size_t len,
                                      cudaStream_t stream) const {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
size_t HashTable<KeyType, ValType>::get_size(cudaStream_t stream) const {
  /* size variable on Host and device, total capacity of the hashtable */
  size_t container_size;

  const size_t hash_capacity = container_->size();

  /* grid_size and allocating/initializing variable on dev, launching kernel*/
  const int grid_size = (hash_capacity - 1) / BLOCK_SIZE_ + 1;

  CK_CUDA(cudaMemsetAsync(d_container_size_, 0, sizeof(size_t), stream));
  size_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, hash_capacity, d_container_size_,
                                                     empty_key);
  CK_CUDA(cudaMemcpyAsync(&container_size, d_container_size_, sizeof(size_t),
                          cudaMemcpyDeviceToHost, stream));
  CK_CUDA(cudaStreamSynchronize(stream));

  return container_size;
}

template <typename KeyType, typename ValType>
size_t HashTable<KeyType, ValType>::get_value_head(cudaStream_t stream) const {
  size_t counter;
  CK_CUDA(cudaMemcpyAsync(&counter, d_counter_, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  CK_CUDA(cudaStreamSynchronize(stream));
  return counter;
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::dump(KeyType* d_key, ValType* d_val, size_t* d_dump_counter,
                                       cudaStream_t stream) const {
  size_t search_length = static_cast<size_t>(capacity_ / LOAD_FACTOR);
  // Before we call the kernel, set the global counter to 0
  CK_CUDA(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
  // grid size according to the searching length.
  const int grid_size = (search_length - 1) / BLOCK_SIZE_ + 1;
  // dump_kernel: dump bucket container_[0, search_length) to d_key and d_val, and report
  // how many buckets are dumped in d_dump_counter.
  size_t shared_size = sizeof(*d_key) * BLOCK_SIZE_ + sizeof(*d_val) * BLOCK_SIZE_;
  dump_kernel<<<grid_size, BLOCK_SIZE_, shared_size, stream>>>(
      d_key, d_val, container_, 0, search_length, d_dump_counter, empty_key);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::set(const KeyType* d_keys, const ValType* d_vals, size_t len,
                                      cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
size_t HashTable<KeyType, ValType>::get_capacity() const {
  return (container_->size());
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::clear(cudaStream_t stream) {
  container_->clear_async(stream);
  set_value_head(0, stream);
}

template class HashTable<int64_t, size_t>;
template class HashTable<uint32_t, size_t>;

}  // namespace HugeCTR