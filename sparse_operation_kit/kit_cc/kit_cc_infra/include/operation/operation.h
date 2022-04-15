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

#ifndef OPERATION_H
#define OPERATION_H

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "operation/construction_context.h"
#include "operation/op_context.h"
#include "tensor_buffer/general_buffer2.hpp"

namespace SparseOperationKit {
/*
 * This class is the interface to represent an operation
 * used inside embedding layer.
 */
class Operation {
 protected:
  template <typename T>
  using Tensor2 = HugeCTR::Tensor2<T>;
  template <typename T>
  using Tensors2 = HugeCTR::Tensors2<T>;

  ConstructionContext_t base_context() const;

 public:
  explicit Operation(ConstructionContext_t context);
  virtual ~Operation() {}
  void AllocateForwardSpaces();
  virtual void allocate_forward_spaces() = 0;
  void AllocateBackwardSpaces();
  virtual void allocate_backward_spaces() = 0;
  void Forward(const Context_t &replica_context, const bool training);
  virtual void forward(const Context_t &replica_context, const bool training) = 0;
  void Backward(const Context_t &replica_context);
  virtual void backward(const Context_t &replica_context) = 0;

  void set_next(std::shared_ptr<Operation> operation);

  void set_op_name(const std::string &op_name);
  std::string get_op_name() const;

  void DumpToFile(const std::string filepath) const;
  /**
   * by default, operation did not do anything when this function is called.
   * if an operation instance has something needed to be dumped to file,
   * it has to override this virtual function and return true.
   */
  using DumpCallBack = std::function<void(std::ofstream &)> &;
  virtual bool dump(DumpCallBack dump_call_back) const;

  void RestoreFromFile(const std::string filepath);
  /**
   * by default, operation did not do anything when this function is called.
   * if an operation has something needed to be restored from file,
   * it has to override this virtual function.
   */
  virtual void restore(const std::ifstream &filestream);

  void LoadEmbeddingValues(const std::shared_ptr<Tensor> &emb_values);
  /**
   * by default, operation did not do anything when this function is called.
   * if an operation instance has something needed to be modified with embedding values,
   * it has to override this virtual function.
   */
  virtual void load_embedding_values(const std::shared_ptr<Tensor> &emb_values);

  static std::string gen_unique_op_name(const std::string op_name);

 private:
  std::shared_ptr<Operation> next_op_ = nullptr;
  ConstructionContext_t base_context_;
  std::string op_name_;

  static std::unordered_set<std::string> operation_names;
};

using Dispatcher = Operation;

}  // namespace SparseOperationKit

#endif  // OPERATION_H