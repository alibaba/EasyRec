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

#ifndef PREPARE_FUNCTIONS_H
#define PREPARE_FUNCTIONS_H

#include "cuda_runtime_api.h"

namespace SparseOperationKit {

void gen_position_for_indices(const int64_t* indices, const size_t elem_num, int64_t* positions,
                              cudaStream_t stream);

void gen_unique_flags_for_indices(const int64_t* indices, const size_t elem_num, uint32_t* flags,
                                  cudaStream_t stream);

void gen_unique_indexes_for_indices(const uint32_t* flags, const uint32_t* prefix_sums,
                                    const size_t elem_num, size_t* indexes,
                                    uint32_t* num_of_uniques, cudaStream_t stream);

}  // namespace SparseOperationKit

#endif  // PREPARE_FUNCTIONS_H