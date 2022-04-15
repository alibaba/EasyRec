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

#include <cmath>

#include "common/include/backward_functions.cuh"
#include "common/include/backward_functions.h"

namespace SparseOperationKit {

template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void expand_input_grad_kernel(const size_t global_batch_size, const size_t slot_num,
                                         const size_t embedding_vec_size,
                                         const TypeKey *replica_row_offset, const TypeEmbeddingComp *wgrad,
                                         TypeEmbeddingComp *replica_input_grad) {
  size_t bid = blockIdx.x;   // each block corresponding to one sample
  size_t tid = threadIdx.x;  // each thread corresponding to one element in the wgrad

  if (bid < global_batch_size && tid < embedding_vec_size) {
    for (size_t i = 0; i < slot_num; i++) {
      size_t row_index = bid * slot_num + i;  // row-index in wgrad
      TypeKey value_offset = replica_row_offset[row_index];
      TypeKey feature_num = replica_row_offset[row_index + 1] - value_offset;

      // in a slot
      for (size_t j = 0; j < feature_num; j++) {
        size_t target_index = value_offset + j;
        replica_input_grad[target_index * embedding_vec_size + tid] =
            wgrad[row_index * embedding_vec_size + tid];
      }
    }  // for i in slot_num
  }    // if bid < global_batch_size && tid < embedding_vec_size
}

template <typename TypeKey, typename TypeEmbeddingComp>
void expand_input_grad(const size_t global_batch_size, const size_t slot_num,
                       const size_t embedding_vec_size, const TypeKey *replica_row_offset,
                       const TypeEmbeddingComp *wgrad, TypeEmbeddingComp *replica_input_grad, cudaStream_t stream) {
  const size_t grid_size = global_batch_size;
  const size_t block_size = embedding_vec_size;
  expand_input_grad_kernel<<<grid_size, block_size, 0, stream>>>(
      global_batch_size, slot_num, embedding_vec_size, replica_row_offset, wgrad,
      replica_input_grad);
}

template void expand_input_grad(const size_t global_batch_size, const size_t slot_num,
                                const size_t embedding_vec_size, const int64_t *replica_row_offset,
                                const float *wgrad, float *replica_input_grad, cudaStream_t stream);
template void expand_input_grad(const size_t global_batch_size, const size_t slot_num,
                                const size_t embedding_vec_size, const int64_t *replica_row_offset,
                                const __half *wgrad, __half *replica_input_grad, cudaStream_t stream);

template <typename TypeEmbeddingComp>
void backward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeEmbeddingComp *top_grad, TypeEmbeddingComp *wgrad,
                  cudaStream_t stream) {
  HugeCTR::backward_sum(batch_size, slot_num, embedding_vec_size, top_grad, wgrad, stream);
}

template void backward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                           const float *top_grad, float *wgrad, cudaStream_t stream);
template void backward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                           const __half *top_grad, __half *wgrad, cudaStream_t stream);

template <typename TypeKey, typename TypeEmbeddingComp>
void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                   const TypeKey *row_offset, const TypeEmbeddingComp *top_grad,
                   TypeEmbeddingComp *wgrad, cudaStream_t stream) {
  HugeCTR::backward_mean(batch_size, slot_size, embedding_vec_size, row_offset, top_grad, wgrad,
                         stream);
}

template void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                            const int64_t *row_offset, const float *top_grad, float *wgrad,
                            cudaStream_t stream);
template void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                            const int64_t *row_offset, const __half *top_grad, __half *wgrad,
                            cudaStream_t stream);

}  // namespace SparseOperationKit
