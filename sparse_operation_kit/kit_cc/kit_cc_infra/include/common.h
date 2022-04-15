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

#ifndef COMMON_H
#define COMMON_H

#include <nccl.h>

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace SparseOperationKit {

#define ErrorBase (std::string(__FILE__) + ":" + std::to_string(__LINE__) + " ")

#define CK_NCCL(cmd)                                                            \
  do {                                                                          \
    ncclResult_t r = (cmd);                                                     \
    if (r != ncclSuccess) {                                                     \
      throw std::runtime_error(ErrorBase + std::string(ncclGetErrorString(r))); \
    }                                                                           \
  } while (0)

#define CK_CUDA(cmd)                                                            \
  do {                                                                          \
    cudaError_t r = (cmd);                                                      \
    if (r != cudaSuccess) {                                                     \
      throw std::runtime_error(ErrorBase + std::string(cudaGetErrorString(r))); \
    }                                                                           \
  } while (0)

#define CK_CUDA_MSG(cmd, msg)                                                                      \
  do {                                                                                             \
    cudaError_t r = (cmd);                                                                         \
    if (r != cudaSuccess) {                                                                        \
      throw std::runtime_error(ErrorBase + std::string(msg) + std::string(cudaGetErrorString(r))); \
    }                                                                                              \
  } while (0)

#define CK_CURAND(cmd)                                         \
  do {                                                         \
    curandStatus_t r = (cmd);                                  \
    if (r != CURAND_STATUS_SUCCESS) {                          \
      throw std::runtime_error(ErrorBase + std::to_string(r)); \
    }                                                          \
  } while (0)

#define CK_CUSPARSE(cmd)                                                                \
  do {                                                                                  \
    cusparseStatus_t error = (cmd);                                                     \
    if (error != CUSPARSE_STATUS_SUCCESS) {                                             \
      throw std::runtime_error(ErrorBase + std::string(cusparseGetErrorString(error))); \
    }                                                                                   \
  } while (0)

#define CK_MPI(cmd)                                                                      \
  do {                                                                                   \
    auto retval = (cmd);                                                                 \
    if (MPI_SUCCESS != retval) {                                                         \
      char estring[MPI_MAX_ERROR_STRING];                                                \
      int estring_len = 0;                                                               \
      MPI_Error_string(retval, estring, &estring_len);                                   \
      throw std::runtime_error(ErrorBase + "MPI failed due to " + std::string(estring)); \
    }                                                                                    \
  } while (0)

#define REQUIRES_TRUE(condition, msg)                              \
  do {                                                             \
    if (!(condition)) throw std::runtime_error(ErrorBase + (msg)); \
  } while (0)

namespace {
inline std::string filter_path(const std::string& path) {
  auto find_str = [](const std::string input, const char* pattern) {
    std::regex reg(pattern);
    std::smatch result;
    if (std::regex_search(input, result, reg))
      return std::string(result.str());
    else
      throw std::runtime_error(ErrorBase + "Filtering path faild.");
  };
  constexpr char pattern[] = "sparse_operation_kit.*$";
  return find_str(path, pattern);
}
}  // anonymous namespace

#define MESSAGE(msg)                                                                               \
  do {                                                                                             \
    std::cout.setf(std::ios::right, std::ios::adjustfield);                                        \
    std::time_t time_instance = std::time(0);                                                      \
    const std::string time_instance_str = std::to_string(time_instance);                           \
    std::tm* time_now = std::localtime(&time_instance);                                            \
    std::cout << time_now->tm_year + 1900 << "-" << std::setfill('0') << std::setw(2)              \
              << std::to_string(1 + time_now->tm_mon) << "-" << std::setfill('0') << std::setw(2)  \
              << std::to_string(time_now->tm_mday) << " " << std::setfill('0') << std::setw(2)     \
              << std::to_string(time_now->tm_hour) << ":" << std::setfill('0') << std::setw(2)     \
              << std::to_string(time_now->tm_min) << ":" << std::setfill('0') << std::setw(2)      \
              << std::to_string(time_now->tm_sec) << "."                                           \
              << time_instance_str.substr(time_instance_str.size() - 6, time_instance_str.size())  \
              << ": I " << filter_path(__FILE__) << ":" << __LINE__ << "] " << (msg) << std::endl; \
  } while (0)

void ncclUniqueId_to_string(const ncclUniqueId& uniqueId, std::string& uniqueId_s);
void string_to_ncclUniqueId(const std::string& uniqueId_s, ncclUniqueId& uniqueId);
void ncclUniqueId_to_int(const ncclUniqueId& uniqueId, int* uniqueId_num);
void int_to_ncclUniqueId(const int32_t* uniqueId_num, ncclUniqueId& uniqueId);

enum class CombinerType { Sum, Mean };
enum class OptimizerType { Adam };
enum class ParamType : uint32_t { RawParam = 0 }; 
enum class DataType : uint32_t {Unknown = 0,
                                Float32 = 1, 
                                Half = 2, Float16 = 2,
                                Int64 = 3,
                                Uint64 = 4,
                                Int32 = 5,
                                Uint32 = 6};
template <typename T> 
DataType DType();

template <DataType dtype> 
struct TypeImpl;

template <DataType dtype>
using Type = typename TypeImpl<dtype>::type;

std::string DataTypeString(DataType dtype);

extern const std::unordered_map<std::string, CombinerType> CombinerMap;
extern const std::unordered_map<std::string, OptimizerType> OptimizerMap;
using optimizer_hyper_params = std::unordered_map<std::string, float>;

class OptimizerHyperParamsHandler {
 public:
  static std::shared_ptr<OptimizerHyperParamsHandler> create(optimizer_hyper_params&& hyper_params);

  float get_hyper_param(const std::string param_name) const;

 private:
  std::unordered_map<std::string, float> hyper_params_;
  explicit OptimizerHyperParamsHandler(optimizer_hyper_params&& hyper_params);
};
using OptimizerHyperParamsHandler_t = std::shared_ptr<OptimizerHyperParamsHandler>;

template <typename type>
void find_item_in_map(const std::unordered_map<std::string, type>& map, const std::string& key,
                      type& value) {
  auto it = map.find(key);
  if (it == map.end()) throw std::runtime_error(ErrorBase + "Count not find " + key + " in map.");
  value = it->second;
}

std::vector<std::string> str_split(const std::string& input_s, const std::string& pattern);
std::string strs_concat(const std::vector<std::string>& str_vector, const std::string& connection);
int32_t string2num(const std::string& str);

bool file_exist(const std::string filename);
void delete_file(const std::string filename);

template <typename T>
void check_numerics(const T* data, uint32_t size, cudaStream_t& stream);

}  // namespace SparseOperationKit

namespace HugeCTR {

class CudaDeviceContext {
  int32_t original_device_;

 public:
  CudaDeviceContext();
  CudaDeviceContext(int32_t device);
  ~CudaDeviceContext() noexcept(false);
  void set_device(int32_t device) const;
};

}  // namespace HugeCTR

#endif  // COMMON_H