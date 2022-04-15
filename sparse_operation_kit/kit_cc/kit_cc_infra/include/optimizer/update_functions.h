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

#ifndef UPDATE_FUNCTIONS_H
#define UPDATE_FUNCTIONS_H

#include <cuda_runtime_api.h>

namespace SparseOperationKit {

template <typename KeyT, typename ValueT>
cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes, const KeyT* d_keys_in,
                      KeyT* d_keys_out, const ValueT* d_values_in, ValueT* d_values_out,
                      size_t num_items, int32_t begin_bit = 0, int32_t end_bit = sizeof(KeyT) * 8,
                      cudaStream_t stream = 0, bool debug_synchronous = false);

template <typename InputIteratorT, typename OutputIteratorT>
cudaError_t InclusiveSum(void* d_temp_storage, size_t& temp_storage_bytes, InputIteratorT d_in,
                         OutputIteratorT d_out, size_t num_items, cudaStream_t stream = 0,
                         bool debug_synchronous = false);

/*This function is used to update Adam momentums globally*/
void opt_adam_update_states_global(const size_t* unique_indexes, const int64_t* sorted_position,
                                   const int64_t* sorted_indices, const uint32_t unique_indices_num,
                                   const float* gradients, const size_t emb_vec_size,
                                   const float beta1, const float beta2, float* m, float* v,
                                   cudaStream_t stream);

/*This function is used to update Adam parameters globally*/
void opt_adam_update_param_global(const size_t emb_vec_size, const size_t max_vocabulary_size,
                                  const float alpha_t, const float beta1, const float beta2,
                                  const float epsilon, float* m, float* v, float* embedding_table,
                                  cudaStream_t stream);

}  // namespace SparseOperationKit

#endif  // UPDATE_FUNCTIONS_H