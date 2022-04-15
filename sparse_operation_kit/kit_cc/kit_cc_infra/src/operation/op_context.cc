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

#include "operation/op_context.h"

#include "common.h"
#include "tensor_buffer/tensor2_wrapper.h"

namespace SparseOperationKit {

Context_t Context::create(const size_t global_replica_id) {
  return Context_t(new Context(global_replica_id));
}

Context::Context(const size_t global_replica_id) : global_replica_id_(global_replica_id) {}

std::shared_ptr<Tensor>& Context::input(const std::string name) {
  auto iter = tensors_.find(name);
  if (iter == tensors_.end()) throw std::runtime_error(ErrorBase + "No input named as " + name);
  return iter->second;
}

std::shared_ptr<Tensor>& Context::output(const std::string name) {
  auto iter = tensors_.find(name);
  if (iter == tensors_.end()) throw std::runtime_error(ErrorBase + "No output named as " + name);
  return iter->second;
}

void Context::set_input(const std::string name, const std::shared_ptr<Tensor>& tensor,
                        const bool overwrite) {
  try {
    if (!tensor) throw std::runtime_error("Invalid tensor buffer, whose name is " + name);
    auto iter = tensors_.find(name);
    if (iter != tensors_.end()) {
      if (overwrite) iter->second = tensor;
    } else {
      tensors_.emplace(std::make_pair(name, tensor));
    }
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
  return;
}

void Context::set_output(const std::string name, const std::shared_ptr<Tensor>& tensor,
                         const bool overwrite) {
  try {
    if (!tensor) throw std::runtime_error("Invalid tensor buffer whose name is " + name);
    auto iter = tensors_.find(name);
    if (iter != tensors_.end()) {
      if (overwrite) iter->second = tensor;
    } else {
      tensors_.emplace(std::make_pair(name, tensor));
    }
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
  return;
}

template <typename T>
void Context::set_output(const std::string name, const HugeCTR::Tensor2<T>& tensor,
                         const bool overwrite) {
  try {
    if (!tensor.allocated()) throw std::runtime_error("Invalid tensor whose name is " + name);
    auto iter = tensors_.find(name);
    if (tensors_.end() != iter) {
      if (overwrite)
        iter->second = Tensor2Wrapper<T>::create(const_cast<HugeCTR::Tensor2<T>&>(tensor));
    } else {
      tensors_.emplace(std::make_pair(
          name, Tensor2Wrapper<T>::create(const_cast<HugeCTR::Tensor2<T>&>(tensor))));
    }
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
  return;
}
template void Context::set_output(const std::string name, const HugeCTR::Tensor2<uint64_t>& tensor,
                                  const bool overwrite);
template void Context::set_output(const std::string name, const HugeCTR::Tensor2<int64_t>& tensor,
                                  const bool overwrite);
template void Context::set_output(const std::string name, const HugeCTR::Tensor2<float>& tensor,
                                  const bool overwrite);
template void Context::set_output(const std::string name, const HugeCTR::Tensor2<uint32_t>& tensor,
                                  const bool overwrite);
template void Context::set_output(const std::string name, const HugeCTR::Tensor2<__half>& tensor,
                                  const bool overwrite);

size_t Context::get_global_replica_id() const { return global_replica_id_; }

void Context::record_internal_tensor(const std::string name, std::shared_ptr<Tensor>& tensor,
                                     const bool overwrite) {
  try {
    if (!tensor) throw std::runtime_error("Invalid internal tensor.");
    auto iter = temp_tensors_.find(name);
    if (iter != temp_tensors_.end()) {
      if (overwrite) iter->second = tensor;
    } else {
      temp_tensors_.emplace(std::make_pair(name, tensor));
    }
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
  return;
}

template <typename T>
void Context::record_internal_tensor(const std::string name, HugeCTR::Tensor2<T>& tensor,
                                     const bool overwrite) {
  try {
    if (!tensor.allocated()) throw std::runtime_error("Invalid tensor whose name is " + name);
    auto iter = temp_tensors_.find(name);
    if (temp_tensors_.end() != iter) {
      if (overwrite) iter->second = Tensor2Wrapper<T>::create(tensor);
    } else {
      temp_tensors_.emplace(std::make_pair(name, Tensor2Wrapper<T>::create(tensor)));
    }
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
  return;
}
template void Context::record_internal_tensor(const std::string name,
                                              HugeCTR::Tensor2<size_t>& tensor,
                                              const bool overwrite);
template void Context::record_internal_tensor(const std::string name,
                                              HugeCTR::Tensor2<int64_t>& tensor,
                                              const bool overwrite);

bool Context::has_internal_tensor(const std::string name) const {
  auto iter = temp_tensors_.find(name);
  return (iter != temp_tensors_.end());
}

std::shared_ptr<Tensor>& Context::get_internal_tensor(const std::string name) {
  auto iter = temp_tensors_.find(name);
  if (iter != temp_tensors_.end())
    return iter->second;
  else
    throw std::runtime_error(ErrorBase + "There is no internal tensor whose name is " + name);
}

}  // namespace SparseOperationKit