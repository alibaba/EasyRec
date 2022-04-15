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

set(NCCL_INC_PATHS
    /usr/include
    /usr/local/include
    $ENV{NCCL_DIR}/include
    )

set(NCCL_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{NCCL_DIR}/lib
    )

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_INC_PATHS})
find_library(NCCL_LIBRARIES NAMES nccl PATHS ${NCCL_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES)

if (NCCL_FOUND)
  message(STATUS "Found NCCL    (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARIES)
endif ()
