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

#ifndef BACKWARD_FUNCTIONS_H_
#define BACKWARD_FUNCTIONS_H_

#include <cuda_runtime_api.h>

namespace SparseOperationKit {

template <typename TypeKey, typename TypeEmbeddingComp>
void expand_input_grad(const size_t global_batch_size, const size_t slot_num,
                       const size_t embedding_vec_size, const TypeKey *replica_row_offset,
                       const TypeEmbeddingComp *wgrad, TypeEmbeddingComp *replica_input_grad, cudaStream_t stream);

template <typename TypeEmbeddingComp>
void backward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeEmbeddingComp *top_grad, TypeEmbeddingComp *wgrad, cudaStream_t stream);

template <typename TypeKey, typename TypeEmbeddingComp>
void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                   const TypeKey *row_offset, const TypeEmbeddingComp *top_grad,
                   TypeEmbeddingComp *wgrad, cudaStream_t stream);

}  // namespace SparseOperationKit

#endif  // BACKWARD_FUNCTIONS_H_