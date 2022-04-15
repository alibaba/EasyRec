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

cmake_minimum_required(VERSION 3.8)

message(STATUS "Building unit_test ops.")

include_directories(
    ${PROJECT_SOURCE_DIR}/unit_test/
)

file(GLOB unit_test_src
    ${PROJECT_SOURCE_DIR}/unit_test/cc/*.cc
    ${PROJECT_SOURCE_DIR}/unit_test/cc/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/unit_test/cc/ops/*.cc
)

file(GLOB recompile_for_unit_test
)

add_definitions(-DUNIT_TEST)

add_library(sok_unit_test SHARED ${unit_test_src} ${recompile_for_unit_test})
target_link_libraries(sok_unit_test PUBLIC sparse_operation_kit)

set_target_properties(sok_unit_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sok_unit_test PROPERTIES CUDA_ARCHITECTURES OFF)