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

#include "parameters/raw_state.h"

#include "common.h"
#include "tensor_buffer/tensor2_wrapper.h"

namespace SparseOperationKit {

std::shared_ptr<RawStates> RawStates::create(
    const std::vector<size_t> shape, const std::shared_ptr<ResourcesManager>& resource_mgr,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    const std::string initializer) {
  return std::shared_ptr<RawStates>(new RawStates(shape, resource_mgr, buffers, initializer));
}

RawStates::RawStates(
    const std::vector<size_t> shape, const std::shared_ptr<ResourcesManager>& resource_mgr,
    const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
    const std::string initializer)
    : resource_mgr_(resource_mgr), shape_(shape), initializer_(Initializer::Get(initializer)) {
  state_tensors_.reserve(resource_mgr_->get_local_gpu_count());
  state_tensors_interface_.reserve(resource_mgr_->get_local_gpu_count());

  for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
    {
      Tensor2<float> tensor;
      buffers[dev_id]->reserve(shape_, &tensor);
      state_tensors_.push_back(tensor);
      state_tensors_interface_.push_back(Tensor2Wrapper<float>::create(tensor));
    }
  }  // for dev_id in local_gpu_count
}

void RawStates::init(const size_t global_replica_id) {
  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
  if (local_replica_id >= state_tensors_.size())
    throw std::runtime_error(ErrorBase +
                             "local_replica_id is out of the range of state_tensor.size().");

  const auto& local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

  initializer_->fill(state_tensors_interface_[local_replica_id], local_gpu->get_sm_count(),
                     local_gpu->get_variant_curand_gen(), local_gpu->get_stream());

  resource_mgr_->sync_gpu(local_replica_id);
}

std::shared_ptr<Tensor>& RawStates::get_tensor(const size_t local_replica_id) {
  if (local_replica_id >= state_tensors_.size())
    throw std::runtime_error(ErrorBase +
                             "local_replica_id is out of the range of state_tensors.size().");

  return state_tensors_interface_[local_replica_id];
}

}  // namespace SparseOperationKit