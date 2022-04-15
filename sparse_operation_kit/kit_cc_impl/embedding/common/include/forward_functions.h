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

#ifndef EMBEDDING_FUNCTIONS_H_
#define EMBEDDING_FUNCTIONS_H_

#include <cuda_runtime_api.h>
#include <nccl.h>

namespace SparseOperationKit {

// max_smem_size_per_sm >= (global_gpu_count * KEY_WARPS_PER_BLOCK * (sizeof(KeyType) +
// sizeof(uint32_t))) * ITEMS_PER_GPU_PER_WARP
//                            + sizeof(uint32_t) * KEY_WARPS_PER_BLOCK * global_gpu_count

// constexpr size_t ITEMS_PER_GPU_PER_WARP = 64;
constexpr size_t KEY_WARPS_PER_BLOCK = 8;
constexpr size_t EMB_LEN_THRESHOLD = 512;
constexpr size_t EMB_WARPS_PER_BLOCK = 32;

template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                 cudaStream_t stream);

// template <typename TypeHashKey, typename TypeEmbeddingComp>
// void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
//                   const TypeHashKey *row_offset, const size_t *hash_value_index,
//                   const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
//                   cudaStream_t stream);

template <typename TypeKey, typename TypeEmbeddingComp>
void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const TypeKey *row_offset, TypeEmbeddingComp *embedding_feature,
                      cudaStream_t stream);

template <typename Type>
void memset_liner(Type *data, Type start_value, Type stride_value, size_t n, cudaStream_t stream);

template <typename T>
ncclDataType_t GetNCCLType();

}  // namespace SparseOperationKit

#endif  // EMBEDDING_FUNCTIONS_H_