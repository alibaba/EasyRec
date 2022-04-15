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

#ifndef OP_CONTEXT_H
#define OP_CONTEXT_H

#include <memory>
#include <string>
#include <unordered_map>

#include "tensor_buffer/tensor2.hpp"
#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {
/*
 * This class is used to provide context for operation
 * and used to exchange input / output between operations.
 */
class Context {
 public:
  static std::shared_ptr<Context> create(const size_t global_replica_id);

  std::shared_ptr<Tensor>& input(const std::string name);
  std::shared_ptr<Tensor>& output(const std::string name);

  void set_input(const std::string name, const std::shared_ptr<Tensor>& tensor,
                 const bool overwrite = false);

  void set_output(const std::string name, const std::shared_ptr<Tensor>& tensor,
                  const bool overwrite = false);
  template <typename T>
  void set_output(const std::string name, const HugeCTR::Tensor2<T>& tensor,
                  const bool overwrite = false);

  size_t get_global_replica_id() const;

  void record_internal_tensor(const std::string name, std::shared_ptr<Tensor>& tensor,
                              const bool overwrite = false);
  template <typename T>
  void record_internal_tensor(const std::string name, HugeCTR::Tensor2<T>& tensor,
                              const bool overwrite = false);

  bool has_internal_tensor(const std::string name) const;
  std::shared_ptr<Tensor>& get_internal_tensor(const std::string name);

 private:
  explicit Context(const size_t global_replica_id);

  const size_t global_replica_id_;

  // because each EmbeddingLayer owns a Context object rather than each operation,
  // so that a unified spaces is used to hold tensors.
  std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors_;

  // this container is used to hold internal / temp tensors
  std::unordered_map<std::string, std::shared_ptr<Tensor>> temp_tensors_;
};

using Context_t = std::shared_ptr<Context>;

}  // namespace SparseOperationKit

#endif  // OP_CONTEXT_H