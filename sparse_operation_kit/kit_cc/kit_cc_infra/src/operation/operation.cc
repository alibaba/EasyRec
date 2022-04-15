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

#include "operation/operation.h"

#include "common.h"
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

namespace SparseOperationKit {

Operation::Operation(ConstructionContext_t context)
    : base_context_(context), op_name_("sok_operation") {}

void Operation::AllocateForwardSpaces() {
  allocate_forward_spaces();
  if (next_op_) next_op_->AllocateForwardSpaces();
}

void Operation::AllocateBackwardSpaces() {
  allocate_backward_spaces();
  if (next_op_) next_op_->AllocateBackwardSpaces();
}

void Operation::Forward(const Context_t &replica_context, const bool training) {
#ifdef USE_NVTX
  nvtxRangeId_t op_fprop =
      nvtxRangeStartA(std::string("Op: " + get_op_name() + " forward").c_str());
#endif
  forward(replica_context, training);
#ifdef USE_NVTX
  nvtxRangeEnd(op_fprop);
#endif
  if (next_op_) next_op_->Forward(replica_context, training);
}

void Operation::Backward(const Context_t &replica_context) {
  if (next_op_) next_op_->Backward(replica_context);
#ifdef USE_NVTX
  nvtxRangeId_t op_bprop =
      nvtxRangeStartA(std::string("Op: " + get_op_name() + " backward").c_str());
#endif
  backward(replica_context);
#ifdef USE_NVTX
  nvtxRangeEnd(op_bprop);
#endif
}

void Operation::set_next(std::shared_ptr<Operation> operation) {
  if (nullptr == next_op_) {  // next_op_ is invalid
    next_op_ = operation;
    return;
  } else {  // next_op_ is valid, then link it to its next_op
    return next_op_->set_next(operation);
  }
}

ConstructionContext_t Operation::base_context() const { return base_context_; }

std::unordered_set<std::string> Operation::operation_names;

std::string Operation::gen_unique_op_name(const std::string op_name) {
  std::string unique_op_name = op_name;

  while (true) {
    auto iter = operation_names.find(unique_op_name);
    if (operation_names.end() != iter) {  // already exists
      auto name_vec = str_split(unique_op_name, /*pattern=*/"_");
      const int32_t num = string2num(*(name_vec.rbegin()));
      if (-1 == num) {  // not numerical
        unique_op_name = op_name + "_" + std::to_string(1);
      } else {  // numerical
        *(name_vec.rbegin()) = std::to_string(num + 1);
        unique_op_name = strs_concat(name_vec, /*connection=*/"_");
      }
    } else {  // not exists
      operation_names.emplace(unique_op_name);
      break;
    }
  }

  return unique_op_name;
}

void Operation::set_op_name(const std::string &op_name) {
  const std::string &unique_op_name = gen_unique_op_name(op_name);
  op_name_ = unique_op_name;
}

std::string Operation::get_op_name() const { return op_name_; }

void Operation::DumpToFile(const std::string filepath) const {
  try {
    std::function<void(std::ofstream &)> call_back;
    if (dump(call_back)) {
      if (0 == base_context()->get_resource_mgr()->get_worker_id()) {  // chief worker
        const std::string filename = filepath + "/" + get_op_name() + ".file";
        std::ofstream file_stream = std::ofstream(filename, std::ios::binary | std::ios::out);
        call_back(file_stream);
        if (file_stream.is_open()) file_stream.close();
        MESSAGE("Saved operation: " + get_op_name() + " to " + filename);
      } else {  // sub worker
        std::ofstream file_stream;
        call_back(file_stream);
        if (file_stream.is_open()) file_stream.close();
      }
    }

    if (next_op_) next_op_->DumpToFile(filepath);

  } catch (const std::exception &error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
}

bool Operation::dump(DumpCallBack dump_call_back) const {
  // by default, it return false, which means this opeation has nothing to dump.
  return false;
}

void Operation::RestoreFromFile(const std::string filepath) {
  try {
    const std::string filename = filepath + "/" + get_op_name() + ".file";
    if (file_exist(filename)) {
      std::ifstream file_stream = std::ifstream(filename, std::ios::binary | std::ios::out);
      restore(file_stream);
      file_stream.close();
    }

    if (next_op_) next_op_->RestoreFromFile(filepath);
  } catch (const std::exception &error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
}

void Operation::restore(const std::ifstream &filestream) {
  // by default, it does nothing.
}

void Operation::LoadEmbeddingValues(const std::shared_ptr<Tensor> &emb_values) {
  load_embedding_values(emb_values);
  if (next_op_) next_op_->LoadEmbeddingValues(emb_values);
}

void Operation::load_embedding_values(const std::shared_ptr<Tensor> &emb_values) {
  // by default, it does nothing.
}

}  // namespace SparseOperationKit