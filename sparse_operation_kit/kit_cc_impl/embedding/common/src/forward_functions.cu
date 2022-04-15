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

#include "common/include/forward_functions.cuh"
#include "common/include/forward_functions.h"

namespace SparseOperationKit {

template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                 cudaStream_t stream) {
  HugeCTR::forward_sum(batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
                       hash_table_value, embedding_feature, stream);
}

template void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                          const int64_t *row_offset, const size_t *hash_value_index,
                          const float *hash_table_value, float *embedding_feature,
                          cudaStream_t stream);
template void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                          const int64_t *row_offset, const size_t *hash_value_index,
                          const float *hash_table_value, __half *embedding_feature,
                          cudaStream_t stream);

// template <typename TypeHashKey, typename TypeEmbeddingComp>
// void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
//                   const TypeHashKey *row_offset, const size_t *hash_value_index,
//                   const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
//                   cudaStream_t stream) {
//     HugeCTR::forward_mean(batch_size, slot_num, embedding_vec_size,
//                           row_offset, hash_value_index, hash_table_value,
//                           embedding_feature, stream);
// }

// template void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
//                   const long long *row_offset, const size_t *hash_value_index,
//                   const float *hash_table_value, float *embedding_feature,
//                   cudaStream_t stream);

template <typename TypeKey, typename TypeEmbeddingComp>
void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const TypeKey *row_offset, TypeEmbeddingComp *embedding_feature,
                      cudaStream_t stream) {
  HugeCTR::do_forward_scale(batchsize_per_gpu, slot_num, embedding_vec_size, row_offset,
                            embedding_feature, stream);
}

template void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                               const int64_t *row_offset, float *embedding_feature,
                               cudaStream_t stream);
template void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                               const int64_t *row_offset, __half *embedding_feature,
                               cudaStream_t stream);

template <typename Type>
void memset_liner(Type *data, Type start_value, Type stride_value, size_t n, cudaStream_t stream) {
  HugeCTR::memset_liner(data, start_value, stride_value, n, stream);
}

template void memset_liner(size_t *data, size_t start_value, size_t stride_value, size_t n,
                           cudaStream_t stream);

template<> ncclDataType_t GetNCCLType<float>() { return ncclFloat32; }
template<> ncclDataType_t GetNCCLType<__half>() { return ncclHalf; }
template<> ncclDataType_t GetNCCLType<int64_t>() { return ncclInt64; }
template<> ncclDataType_t GetNCCLType<uint32_t>() { return ncclUint32; }

}  // namespace SparseOperationKit