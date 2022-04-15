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

set(NVTX_INC_PATHS
    /usr/local/
    /usr/local/include/
    /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include/
    $ENV{NVTX_DIR}/include/
)

set(NVTX_LIB_PATHS
    /lib/
    /lib64/
    /usr/lib/
    /usr/local/
    /usr/local/lib/
    /usr/local/lib64/
    /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/
    $ENV{NVTX_DIR}/lib
)

find_path(NVTX_INCLUDE_DIR NAMES nvToolsExt.h PATHS ${NVTX_INC_PATHS})
if (NOT NVTX_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find nvtx_include_directories.")
endif()

find_library(NVTX_LIB NAMES nvToolsExt PATHS ${NVTX_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIR NVTX_LIB)

if (NVTX_FOUND)
    message(STATUS "Found NVTX: include: ${NVTX_INCLUDE_DIR}, library: ${NVTX_LIB}")
    mark_as_advanced(NVTX_INCLUDE_DIR NVTX_LIB)
endif()