# 
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Sometimes, the framework does not provide GPU implementations for specific 
# ops, or its implementations are not effecient enough. This folder is to 
# accommodate those customized ops.

cmake_minimum_required(VERSION 3.8)

message(STATUS "Building SOK Operations.")

include_directories(
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/
)

file(GLOB operations_src
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/operations/cc/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/operations/cc/*.cu
)

file(GLOB framework_interface
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/compat/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/compat/kernels/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/compat/ops/*.cc
)

add_library(sparse_operation_kit_compat_ops SHARED ${operations_src} ${framework_interface})
target_link_libraries(sparse_operation_kit_compat_ops PUBLIC ${TF_LINK_FLAGS})

set_target_properties(sparse_operation_kit_compat_ops PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sparse_operation_kit_compat_ops PROPERTIES CUDA_ARCHITECTURES OFF)