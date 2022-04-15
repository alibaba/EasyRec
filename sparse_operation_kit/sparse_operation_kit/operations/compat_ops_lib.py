"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from tensorflow.python.framework import load_library
import os

lib_name = r"libsparse_operation_kit_compat_ops.so"

install_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib"))
paths = [r"/usr/local/lib", install_path]

lib_file = None
for path in paths:
    try:
        file = open(os.path.join(path, lib_name))
        file.close()
        lib_file = os.path.join(path, lib_name)
        break
    except FileNotFoundError:
        continue

if lib_file is None:
    raise FileNotFoundError(f"Could not find {lib_name}")

sok_compat_ops = load_library.load_op_library(lib_file)

#for op in dir(sok_compat_ops):
#    print(op)

unique_op = sok_compat_ops.gpu_unique
unsorted_segment_sum_op = sok_compat_ops.gpu_unsorted_segment_sum