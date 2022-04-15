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

#ifndef RAW_STATE_H
#define RAW_STATE_H

#include "initializer/initializer_interface.h"
#include "parameters/state_interface.h"
#include "resources/manager.h"
#include "tensor_buffer/general_buffer2.hpp"
#include "tensor_buffer/tensor2.hpp"

namespace SparseOperationKit {

class RawStates : public States {
  template <typename T>
  using Tensor2 = HugeCTR::Tensor2<T>;
  template <typename T>
  using Tensors2 = HugeCTR::Tensors2<T>;

  RawStates(
      const std::vector<size_t> shape, const std::shared_ptr<ResourcesManager>& resource_mgr,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      const std::string initializer);

  std::shared_ptr<ResourcesManager> resource_mgr_;

  const std::vector<size_t> shape_;
  std::shared_ptr<Initializer> initializer_;

  // currently, this class only store float states.
  Tensors2<float> state_tensors_;
  std::vector<std::shared_ptr<Tensor>> state_tensors_interface_;

 public:
  static std::shared_ptr<RawStates> create(
      const std::vector<size_t> shape, const std::shared_ptr<ResourcesManager>& resource_mgr,
      const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
      const std::string initializer = "zeros");

  void init(const size_t global_replica_id) override;
  std::shared_ptr<Tensor>& get_tensor(const size_t local_replica_id) override;
};

}  // namespace SparseOperationKit

#endif  // RAW_STATE_H