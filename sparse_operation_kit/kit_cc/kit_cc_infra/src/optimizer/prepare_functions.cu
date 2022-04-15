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

#include "optimizer/prepare_functions.h"

namespace SparseOperationKit {

__global__ void gen_position_for_indices_kernel(const int64_t* indices, const size_t elem_num,
                                                int64_t* positions) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < elem_num) {
    positions[gid] = gid;
  }
}

void gen_position_for_indices(const int64_t* indices, const size_t elem_num, int64_t* positions,
                              cudaStream_t stream) {
  size_t block_size = 64;
  size_t grid_size = (block_size + elem_num - 1) / block_size;
  gen_position_for_indices_kernel<<<grid_size, block_size, 0, stream>>>(indices, elem_num,
                                                                        positions);
}

__global__ void gen_unique_flags_for_indices_kernel(const int64_t* indices, const size_t elem_num,
                                                    uint32_t* flags) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < elem_num; i += stride) {
    int64_t cur_indice = indices[i];
    if (i > 0) {
      int64_t former_indice = indices[i - 1];
      if (cur_indice != former_indice)
        flags[i] = 1;  // it is an unique indice
      else
        flags[i] = 0;  // it is not an unique indice
    } else {
      flags[i] = 1;  // it is an unique indice
    }
  }
}

void gen_unique_flags_for_indices(const int64_t* indices, const size_t elem_num, uint32_t* flags,
                                  cudaStream_t stream) {
  constexpr size_t max_grid_size = 384;
  size_t block_size = 256;
  size_t grid_size = std::min(max_grid_size, (elem_num - 1) / block_size + 1);
  gen_unique_flags_for_indices_kernel<<<grid_size, block_size, 0, stream>>>(indices, elem_num,
                                                                            flags);
}

__global__ void gen_unique_indexes_for_indices_kernel(const uint32_t* flags,
                                                      const uint32_t* prefix_sums,
                                                      const size_t elem_num, size_t* indexes,
                                                      uint32_t* num_of_uniques) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < elem_num; i += stride) {
    uint32_t flag = flags[i];
    if (1 == flag) indexes[prefix_sums[i] - 1] = i;
  }
  if (0 == gid) {
    *num_of_uniques = prefix_sums[elem_num - 1];
    indexes[*num_of_uniques] = elem_num;
  }
}

void gen_unique_indexes_for_indices(const uint32_t* flags, const uint32_t* prefix_sums,
                                    const size_t elem_num, size_t* indexes,
                                    uint32_t* num_of_uniques, cudaStream_t stream) {
  constexpr size_t max_grid_size = 384;
  size_t block_size = 256;
  size_t grid_size = std::min(max_grid_size, (elem_num - 1) / block_size + 1);
  gen_unique_indexes_for_indices_kernel<<<grid_size, block_size, 0, stream>>>(
      flags, prefix_sums, elem_num, indexes, num_of_uniques);
}

}  // namespace SparseOperationKit