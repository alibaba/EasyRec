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
project(SparseOperationKit LANGUAGES CXX CUDA)

message(STATUS "Building Sparse Operation Kit from source.")

set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakes)

find_package(CUDA REQUIRED)
find_package(NCCL REQUIRED)
find_package(TensorFlow REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads)

set(CUDA_SEPARABLE_COMPILATION ON)

# whether use nvtx
option(USE_NVTX "Use nvtx for profiling" OFF)
if (USE_NVTX)
    message(STATUS "Add nvtx for profiling")
    add_definitions(-DUSE_NVTX)
    find_package(NVTX REQUIRED)
endif()

# whether make ops to be asynchronous
option(SOK_ASYNC "Make Ops to be asynchronous" ON)
if (SOK_ASYNC)
    message(STATUS "Make Ops to be asynchronous")
    add_definitions(-DSOK_ASYNC)
endif()

# setting compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

foreach(arch_name ${SM})
    if (arch_name STREQUAL 80 OR 
        arch_name STREQUAL 75 OR 
        arch_name STREQUAL 70 OR 
        arch_name STREQUAL 61 OR 
        arch_name STREQUAL 60)
        list(APPEND cuda_arch_list ${arch_name})
        message(STATUS "Assign GPU architecture (sm=${arch_name})")
    else()
        message(FATAL_ERROR "Unknown or unsupported GPU architecture (set sm=70)")
    endif()
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(${cuda_arch_list_length} EQUAL 0)
    list(APPEND cuda_arch_list "70")
    message(STATUS "Assign default GPU architecture sm=70")
endif()
list(REMOVE_DUPLICATES cuda_arch_list)

foreach(arch_name ${cuda_arch_list})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch_name},code=sm_${arch_name}")
endforeach()

if (${CMAKE_BUILD_TYPE} EQUAL "Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unknown-pragmas")
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -Xcompiler -Wall,-Werror,-Wno-error=cpp")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
else() 
    # -------- set flags for DEBUG mode --- #
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-unknown-pragmas")
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-error=class-memaccess")
    endif()
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -rdc=true -Xcompiler -Wall,-Wno-error=cpp")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcudafe --display_error_number")
    # set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    # ------------------------------------- #
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_COMPILE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGOOGLE_CUDA=1 ${TF_COMPILE_FLAGS}")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# setting install folder
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION /usr/local)

# headers
include_directories(
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/include/
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/third_party/
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/
    ${CUDA_INCLUDE_DIRS}
    ${NCCL_INC_PATHS}
    ${MPI_INCLUDE_PATH}
)


# libs
link_directories(
    ${NCCL_LIB_PATHS}
    ${TF_LINK_DIR}/
)

# code sources related to framework
file(GLOB framework
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/*.cpp
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/kernels/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/ops/*.cpp
    ${PROJECT_SOURCE_DIR}/kit_cc/framework/ops/*.cc
)

# code sources related to the kit infrastructure
file(GLOB kit_infra
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/*.cpp
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/resources/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/parameters/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/parameters/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/hashtable/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/hashtable/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/embeddings/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/embeddings/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/tensor_buffer/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/dispatcher/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/dispatcher/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/initializer/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/initializer/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/operation/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/operation/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/optimizer/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc/kit_cc_infra/src/optimizer/*.cu
)

# code sources for the concrete algorithm implementations of the kit
file(GLOB kit_impl
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/common/src/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/common/src/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/dispatcher/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/dispatcher/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/lookuper/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/lookuper/*.cu
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/operation/*.cc
    ${PROJECT_SOURCE_DIR}/kit_cc_impl/embedding/operation/*.cu
)

file(GLOB test_op
    ${PROJECT_SOURCE_DIR}/Plugin/framework/kernels/test_op.cc
    ${PROJECT_SOURCE_DIR}/Plugin/framework/ops/test_op.cc
)

# build dynamic lib
add_library(sparse_operation_kit SHARED ${framework} ${kit_infra} ${kit_impl})
target_link_libraries(sparse_operation_kit PUBLIC ${TF_LINK_FLAGS} nccl cusparse ${MPI_CXX_LIBRARIES})

if (USE_NVTX)
    target_link_libraries(sparse_operation_kit PUBLIC ${NVTX_LIB})
endif()

set_target_properties(sparse_operation_kit PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sparse_operation_kit PROPERTIES CUDA_ARCHITECTURES OFF)

# compile compat ops
add_subdirectory(kit_cc_impl/operations)

# compile ops for unit test
option(SOK_UNIT_TEST "Compile SOK unit test" ON)
if (SOK_UNIT_TEST)
    message(STATUS "Compile SOK unit test")
    add_subdirectory(unit_test)
endif()