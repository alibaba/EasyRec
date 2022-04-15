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

#include <algorithm>
#include <cub/cub.cuh>

#include "optimizer/update_functions.h"

namespace HugeCTR {}  // namespace HugeCTR

namespace SparseOperationKit {

template <typename KeyT, typename ValueT>
cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes, const KeyT* d_keys_in,
                      KeyT* d_keys_out, const ValueT* d_values_in, ValueT* d_values_out,
                      size_t num_items, int32_t begin_bit, int32_t end_bit, cudaStream_t stream,
                      bool debug_synchronous) {
  return cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                         d_values_in, d_values_out, num_items, begin_bit, end_bit,
                                         stream, debug_synchronous);
}

template cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                               const size_t* d_keys_in, size_t* d_keys_out,
                               const int64_t* d_values_in, int64_t* d_values_out, size_t num_items,
                               int32_t begin_bit, int32_t end_bit, cudaStream_t stream,
                               bool debug_synchronous);
template cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                               const int64_t* d_keys_in, int64_t* d_keys_out,
                               const int64_t* d_values_in, int64_t* d_values_out, size_t num_items,
                               int32_t begin_bit, int32_t end_bit, cudaStream_t stream,
                               bool debug_synchronous);

template <typename InputIteratorT, typename OutputIteratorT>
cudaError_t InclusiveSum(void* d_temp_storage, size_t& temp_storage_bytes, InputIteratorT d_in,
                         OutputIteratorT d_out, size_t num_items, cudaStream_t stream,
                         bool debug_synchronous) {
  return cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items,
                                       stream, debug_synchronous);
}

template cudaError_t InclusiveSum(void* d_temp_storage, size_t& temp_storage_bytes, uint32_t* d_in,
                                  uint32_t* d_out, size_t num_items, cudaStream_t stream,
                                  bool debug_synchronous);

__device__ __forceinline__ float accumulate_gradients(const size_t* unique_indexes,
                                                      const int64_t* sorted_position,
                                                      const size_t emb_vec_size,
                                                      const float* gradients, const size_t bid,
                                                      const size_t tid) {
  size_t repeated_num = unique_indexes[bid + 1] - unique_indexes[bid];

  float summed = 0.0f;
  for (size_t i = 0; i < repeated_num; i++) {
    size_t index = unique_indexes[bid] + i;
    int64_t sort_p = sorted_position[index];
    summed += gradients[sort_p * emb_vec_size + tid];
  }
  return summed;
}

__global__ void opt_adam_update_states_global_kernel(
    const size_t* unique_indexes, const int64_t* sorted_position, const int64_t* sorted_indices,
    const uint32_t unique_indices_num, const float* gradients, const size_t emb_vec_size,
    const float beta1, const float beta2, float* m, float* v) {
  size_t bid = blockIdx.x;
  size_t tid = threadIdx.x;

  if (tid < emb_vec_size && bid < unique_indices_num) {
    float summed =
        accumulate_gradients(unique_indexes, sorted_position, emb_vec_size, gradients, bid, tid);
    size_t index = unique_indexes[bid];
    int64_t indice = sorted_indices[index];
    size_t indice_in_param = indice * emb_vec_size + tid;

    // m = ( m * beta1 + (1 - beta1) * grad ) / beta1
    m[indice_in_param] += (1.0f - beta1) * summed / beta1;
    // v = ( v * beta2 + (1 - beta2) * grad * grad ) / beta2
    v[indice_in_param] += (1.0f - beta2) * (summed * summed) / beta2;
  }
}

void opt_adam_update_states_global(const size_t* unique_indexes, const int64_t* sorted_position,
                                   const int64_t* sorted_indices, const uint32_t unique_indices_num,
                                   const float* gradients, const size_t emb_vec_size,
                                   const float beta1, const float beta2, float* m, float* v,
                                   cudaStream_t stream) {
  size_t block_size = emb_vec_size;
  size_t grid_size = std::max(1u, unique_indices_num);
  opt_adam_update_states_global_kernel<<<grid_size, block_size, 0, stream>>>(
      unique_indexes, sorted_position, sorted_indices, unique_indices_num, gradients, emb_vec_size,
      beta1, beta2, m, v);
}

__global__ void opt_adam_update_param_global_kernel(const size_t emb_vec_size,
                                                    const size_t max_vocabulary_size,
                                                    const float alpha_t, const float beta1,
                                                    const float beta2, const float epsilon,
                                                    float* m, float* v, float* embedding_table) {
  const size_t stride = blockDim.x * gridDim.x;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gid; i < max_vocabulary_size * emb_vec_size; i += stride) {
    float mi = m[i] * beta1;
    float vi = v[i] * beta2;
    m[i] = mi;
    v[i] = vi;

    float weight_diff = (-alpha_t * mi) / (sqrtf(vi) + epsilon);
    embedding_table[i] += weight_diff;
  }
}

void opt_adam_update_param_global(const size_t emb_vec_size, const size_t max_vocabulary_size,
                                  const float alpha_t, const float beta1, const float beta2,
                                  const float epsilon, float* m, float* v, float* embedding_table,
                                  cudaStream_t stream) {
  opt_adam_update_param_global_kernel<<<1024, 256, 0, stream>>>(
      emb_vec_size, max_vocabulary_size, alpha_t, beta1, beta2, epsilon, m, v, embedding_table);
}

}  // namespace SparseOperationKit