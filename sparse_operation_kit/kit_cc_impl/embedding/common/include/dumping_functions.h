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

#ifndef DUMPING_FUNCTIONS_H
#define DUMPING_FUNCTIONS_H

#include <cuda_runtime_api.h>

#include "parameters/param_interface.h"
#include "resources/manager.h"

namespace SparseOperationKit {

void get_hash_value(size_t count, size_t embedding_vec_size, const size_t *value_index,
                    const float *embedding_table, float *value_retrieved, cudaStream_t stream);

// distribute keys to GPU based on key % GPU_NUM
template <typename KeyType>
void save_params_helper(const std::shared_ptr<ParamInterface> &param,
                        const std::shared_ptr<ResourcesManager> &resource_mgr,
                        std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
                        size_t &num_total_keys);

template <typename KeyType>
void restore_params_helper(std::shared_ptr<ParamInterface> &param,
                           const std::shared_ptr<ResourcesManager> &resource_mgr,
                           const std::shared_ptr<Tensor> &keys,
                           const std::shared_ptr<Tensor> &embedding_values,
                           const size_t num_total_keys);
template <typename KeyType>
void generate_dummy_keys(KeyType *d_keys, const size_t num_keys, const size_t global_replica_id,
                         const size_t global_gpu_count, cudaStream_t stream);

}  // namespace SparseOperationKit

#endif  // DUMPING_FUNCTIONS_H