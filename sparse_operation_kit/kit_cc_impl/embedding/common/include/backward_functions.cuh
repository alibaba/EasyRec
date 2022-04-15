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

#ifndef BACKWARD_FUNCTIONS_CUH_
#define BACKWARD_FUNCTIONS_CUH_

#include "common.cuh"

namespace HugeCTR {

// backward kernel function: for combiner=sum
template <typename TypeEmbeddingComp>
__global__ void backward_sum_kernel(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                                    const TypeEmbeddingComp *top_grad, TypeEmbeddingComp *wgrad) {
  size_t tid = threadIdx.x;
  size_t bid = blockIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (size_t i = 0; i < slot_num; i++) {
      size_t feature_index = (size_t)(bid * slot_num + i) * embedding_vec_size + tid;
      wgrad[feature_index] = top_grad[feature_index];
    }
  }
}

template <typename TypeEmbeddingComp>
void backward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeEmbeddingComp *top_grad, TypeEmbeddingComp *wgrad,
                  cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size = embedding_vec_size;
  backward_sum_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num,
                                                            embedding_vec_size, top_grad, wgrad);
}

// backward kernel function: for combiner=mean
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void backward_mean_kernel(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                                     const TypeKey *row_offset, const TypeEmbeddingComp *top_grad,
                                     TypeEmbeddingComp *wgrad) {
  size_t bid = blockIdx.x;
  size_t tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (size_t i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      size_t value_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      float scaler = 1.0f;
      if (value_num > 1) {
        scaler = 1.0f / value_num;  // partial derivatice of MEAN
      }

      size_t feature_index = feature_row_index * embedding_vec_size + tid;
      float g = TypeConvertFunc<float, TypeEmbeddingComp>::convert(top_grad[feature_index]);
      g *= scaler;
      wgrad[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(g);
    }
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                   const TypeKey *row_offset, const TypeEmbeddingComp *top_grad,
                   TypeEmbeddingComp *wgrad, cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size = embedding_vec_size;
  backward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size, slot_size, embedding_vec_size, row_offset, top_grad, wgrad);
}

}  // namespace HugeCTR

#endif  // BACKWARD_FUNCTIONS_CUH_