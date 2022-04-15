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

#include "common.h"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace SparseOperationKit {

template <> DataType DType<float>() { return DataType::Float32; }
template <> DataType DType<__half>() { return DataType::Float16; }
template <> DataType DType<int64_t>() { return DataType::Int64; }
template <> DataType DType<uint64_t>() { return DataType::Uint64; }
template <> DataType DType<int32_t>() { return DataType::Int32; }
template <> DataType DType<uint32_t>() { return DataType::Uint32; }
 
template <> struct TypeImpl<DataType::Float32> { using type = float; };
template <> struct TypeImpl<DataType::Float16> { using type = __half; };
template <> struct TypeImpl<DataType::Int64> { using type = int64_t; };
template <> struct TypeImpl<DataType::Uint64> { using type = uint64_t; };
template <> struct TypeImpl<DataType::Int32> { using type = int32_t; };
template <> struct TypeImpl<DataType::Uint32> { using type = uint32_t; };

std::string DataTypeString(DataType dtype) {
  switch (dtype) {
    case DataType::Float32: { return "Float32"; }
    case DataType::Float16: { return "Float16"; }
    case DataType::Int64: { return "Int64"; }
    case DataType::Uint64: { return "Uint64"; }
    case DataType::Int32: { return "Int32"; }
    case DataType::Uint32: { return "Uint32"; }
    default: {
      throw std::runtime_error(ErrorBase + "Unknown dtype.");
    }
  }
}

void ncclUniqueId_to_string(const ncclUniqueId& uniqueId, std::string& uniqueId_s) {
  uniqueId_s = std::string(uniqueId.internal, NCCL_UNIQUE_ID_BYTES);
}

void string_to_ncclUniqueId(const std::string& uniqueId_s, ncclUniqueId& uniqueId) {
  if (uniqueId_s.size() != NCCL_UNIQUE_ID_BYTES) {
    throw std::runtime_error(ErrorBase +
                             "The size of string is not equal to NCCL_UNIQUE_ID_BYTES.");
  }
  std::memcpy(uniqueId.internal, uniqueId_s.data(), NCCL_UNIQUE_ID_BYTES);
}

void ncclUniqueId_to_int(const ncclUniqueId& uniqueId, int32_t* uniqueId_num) {
  std::memcpy(uniqueId_num, uniqueId.internal, NCCL_UNIQUE_ID_BYTES);
}
void int_to_ncclUniqueId(const int32_t* uniqueId_num, ncclUniqueId& uniqueId) {
  std::memcpy(uniqueId.internal, uniqueId_num, NCCL_UNIQUE_ID_BYTES);
}

const std::unordered_map<std::string, CombinerType> CombinerMap = {{"sum", CombinerType::Sum},
                                                                   {"mean", CombinerType::Mean}};
const std::unordered_map<std::string, OptimizerType> OptimizerMap = {{"Adam", OptimizerType::Adam}};

std::shared_ptr<OptimizerHyperParamsHandler> OptimizerHyperParamsHandler::create(
    optimizer_hyper_params&& hyper_params) {
  return std::shared_ptr<OptimizerHyperParamsHandler>(
      new OptimizerHyperParamsHandler(std::move(hyper_params)));
}

OptimizerHyperParamsHandler::OptimizerHyperParamsHandler(optimizer_hyper_params&& hyper_params)
    : hyper_params_(hyper_params) {}

float OptimizerHyperParamsHandler::get_hyper_param(const std::string param_name) const {
  const auto iter = hyper_params_.find(param_name);
  if (hyper_params_.end() == iter)
    throw std::runtime_error(ErrorBase + "Cannot find hyper param whose name is: " + param_name);
  return iter->second;
}

std::vector<std::string> str_split(const std::string& input_s, const std::string& pattern) {
  std::regex re(pattern);
  std::sregex_token_iterator p(input_s.begin(), input_s.end(), re, -1);
  std::sregex_token_iterator end;
  std::vector<std::string> result;
  while (p != end) result.emplace_back(*p++);
  return result;
}

std::string strs_concat(const std::vector<std::string>& str_vector, const std::string& connection) {
  std::string result = "";
  for (size_t i = 0; i < str_vector.size() - 1; ++i) result += (str_vector[i] + connection);
  result += *(str_vector.rbegin());
  return result;
}
int32_t string2num(const std::string& str) {
  int32_t result = -1;
  std::stringstream ss(str);
  return (!(ss >> result) ? -1 : result);
}

void delete_file(const std::string filename) {
  try {
    if (0 != std::remove(filename.c_str()))
      throw std::runtime_error("Delete " + filename + " failed.");
  } catch (const std::exception& error) {
    throw std::runtime_error(ErrorBase + error.what());
  }
}

bool file_exist(const std::string filename) {
  const std::ifstream file(filename);
  return file.good();
}

}  // namespace SparseOperationKit

namespace HugeCTR {

CudaDeviceContext::CudaDeviceContext() { CK_CUDA(cudaGetDevice(&original_device_)); }

CudaDeviceContext::CudaDeviceContext(int32_t device) : CudaDeviceContext() {
  if (device != original_device_) {
    set_device(device);
  }
}

CudaDeviceContext::~CudaDeviceContext() noexcept(false) { set_device(original_device_); }

void CudaDeviceContext::set_device(int32_t device) const { CK_CUDA(cudaSetDevice(device)); }

}  // namespace HugeCTR