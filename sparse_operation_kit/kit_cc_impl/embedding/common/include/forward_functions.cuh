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

#ifndef EMBEDDING_FUNCTIONS_CUH_
#define EMBEDDING_FUNCTIONS_CUH_

#include "common.cuh"

namespace SparseOperationKit {

// forward kernel funcion: No reduction
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_kernel(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                               const TypeKey *row_offset, const size_t *hash_value_index,
                               const float *hash_table_value,
                               TypeEmbeddingComp *embedding_feature) {
  size_t bid = blockIdx.x;   // each block corresponding to one sample
  size_t tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (size_t i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      TypeKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      for (size_t j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        TypeEmbeddingComp embedding_value =
            hash_table_value[value_index * embedding_vec_size + tid];
        embedding_feature[(value_offset + j) * embedding_vec_size + tid] =
            HugeCTR::TypeConvertFunc<TypeEmbeddingComp, float>(embedding_value);
      }  // for j in feature_num

    }  // for i in slot_num
  }
}

// No reduction
template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
             const TypeHashKey *row_offset, const size_t *hash_value_index,
             const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
             cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size =
      embedding_vec_size;  // each thread corresponds to one element in an embedding vector
  forward_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num, embedding_vec_size,
                                                       row_offset, hash_value_index,
                                                       hash_table_value, embedding_feature);
}

}  // namespace SparseOperationKit

namespace HugeCTR {

// forward kernel funcion: for both combiner=sum and combiner=mean
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_sum_kernel(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                                   const TypeKey *row_offset, const size_t *hash_value_index,
                                   const float *hash_table_value,
                                   TypeEmbeddingComp *embedding_feature) {
  size_t bid = blockIdx.x;   // each block corresponding to one sample
  size_t tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (size_t i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      TypeKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      float sum = 0.0f;

      // reduce in a slot
      for (size_t j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum += hash_table_value[value_index * embedding_vec_size + tid];
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum);
    }
  }
}

// do sum reduction
template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                 cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size =
      embedding_vec_size;  // each thread corresponds to one element in an embedding vector
  forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num, embedding_vec_size,
                                                           row_offset, hash_value_index,
                                                           hash_table_value, embedding_feature);
}

// // forward kernel funcion: for combiner=mean
// template <typename TypeKey, typename TypeEmbeddingComp>
// __global__ void forward_mean_kernel(int batch_size, int slot_num, int embedding_vec_size,
//                                     const TypeKey *row_offset, const size_t *hash_value_index,
//                                     const float *hash_table_value,
//                                     TypeEmbeddingComp *embedding_feature) {
//   int bid = blockIdx.x;   // each block corresponding to one sample
//   int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

//   if (bid < batch_size && tid < embedding_vec_size) {
//     for (int i = 0; i < slot_num; i++) {
//       int feature_row_index = bid * slot_num + i;
//       TypeKey value_offset = row_offset[feature_row_index];
//       int feature_num =
//           row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

//       float sum = 0.0f;

//       // reduce in a slot
//       for (int j = 0; j < feature_num; j++) {
//         size_t value_index = hash_value_index[value_offset + j];
//         sum += hash_table_value[value_index * embedding_vec_size + tid];
//       }

//       float scaler = 1.0f;
//       if (feature_num > 1) {
//         scaler = 1.0f / feature_num;
//       }

//       // store the embedding vector
//       embedding_feature[feature_row_index * embedding_vec_size + tid] =
//           TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum * scaler);
//     }
//   }
// }

// template <typename TypeHashKey, typename TypeEmbeddingComp>
// void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
//                   const TypeHashKey *row_offset, const size_t *hash_value_index,
//                   const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
//                   cudaStream_t stream) {
//   const size_t grid_size = batch_size;
//   const size_t block_size = embedding_vec_size;
//   forward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
//       batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
//       embedding_feature);
// }

template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_scale_kernel(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                                     const TypeKey *row_offset,
                                     TypeEmbeddingComp *embedding_feature) {
  size_t bid = blockIdx.x;
  size_t tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (size_t i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      size_t feature_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      size_t feature_index = feature_row_index * embedding_vec_size + tid;
      float feature =
          TypeConvertFunc<float, TypeEmbeddingComp>::convert(embedding_feature[feature_index]);
      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / (float)feature_num;
      }

      embedding_feature[feature_index] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(feature * scaler);
    }
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const TypeKey *row_offset, TypeEmbeddingComp *embedding_feature,
                      cudaStream_t stream) {
  const size_t grid_size = batchsize_per_gpu;
  const size_t block_size = embedding_vec_size;
  forward_scale_kernel<<<grid_size, block_size, 0, stream>>>(
      batchsize_per_gpu, slot_num, embedding_vec_size, row_offset, embedding_feature);
};

// get hash_value by value_index from hash_table_value matrix
template <typename TypeValueIndex>
__global__ void get_hash_value_kernel(size_t count, size_t embedding_vec_size,
                                      const TypeValueIndex *value_index,
                                      const float *hash_table_value, float *value_retrieved) {
  size_t tid = threadIdx.x;
  size_t bid = blockIdx.x;

  if (bid < count && tid < embedding_vec_size) {
    size_t index = value_index[bid];  // row number in the hash_table_value matrix
    value_retrieved[bid * embedding_vec_size + tid] =
        hash_table_value[index * embedding_vec_size + tid];
  }
}

// memset liner data to the buffer
template <typename Type>
__global__ void memset_liner_kernel(Type *data, const Type start_value, const Type stride_value,
                                    size_t n) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    data[gid] = start_value + gid * stride_value;
  }
}

template <typename Type>
void memset_liner(Type *data, Type start_value, Type stride_value, size_t n, cudaStream_t stream) {
  const size_t block_size = 256;
  const size_t grid_size = (n + block_size - 1) / block_size;

  memset_liner_kernel<<<grid_size, block_size, 0, stream>>>(data, start_value, stride_value, n);
}

}  // namespace HugeCTR

#endif  // EMBEDDING_FUNCTIONS_CUH_