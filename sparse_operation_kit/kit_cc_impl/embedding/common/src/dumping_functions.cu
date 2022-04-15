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

#include "common/include/dumping_functions.h"
#include "common/include/forward_functions.cuh"

namespace SparseOperationKit {

void get_hash_value(size_t count, size_t embedding_vec_size, const size_t *value_index,
                    const float *embedding_table, float *value_retrieved, cudaStream_t stream) {
  const size_t block_size = embedding_vec_size;
  const size_t grid_size = count;

  HugeCTR::get_hash_value_kernel<<<grid_size, block_size, 0, stream>>>(
      count, embedding_vec_size, value_index, embedding_table, value_retrieved);
}

// keys are [0, num_keys)
template <typename KeyType>
__global__ void generate_dummy_keys_kernel(KeyType *d_keys, const size_t num_keys,
                                           const size_t global_replica_id,
                                           const size_t global_gpu_count) {
  const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < num_keys; i += stride) {
    // d_keys[i] = static_cast<KeyType>(global_gpu_count * i + global_replica_id);
    d_keys[i] = static_cast<KeyType>(i);
  }
}
template <typename KeyType>
void generate_dummy_keys(KeyType *d_keys, const size_t num_keys, const size_t global_replica_id,
                         const size_t global_gpu_count, cudaStream_t stream) {
  constexpr size_t block_size = 256ul;
  const size_t grid_size = (num_keys + block_size - 1) / block_size;
  generate_dummy_keys_kernel<<<grid_size, block_size, 0, stream>>>(
      d_keys, num_keys, global_replica_id, global_gpu_count);
}

template void generate_dummy_keys(int64_t *d_keys, const size_t num_keys,
                                  const size_t global_replica_id, const size_t global_gpu_count,
                                  cudaStream_t stream);
template void generate_dummy_keys(uint32_t *d_keys, const size_t num_keys,
                                  const size_t global_replica_id, const size_t global_gpu_count,
                                  cudaStream_t stream);

}  // namespace SparseOperationKit