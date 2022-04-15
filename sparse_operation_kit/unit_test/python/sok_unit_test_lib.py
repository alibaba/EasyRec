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

lib_name = r"libsok_unit_test.so"

paths = [r"/usr/local/lib"]

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
    raise FileNotFoundError("Could not find %s" %lib_name)

plugin_unit_test_ops = load_library.load_op_library(lib_file)

# for op in dir(plugin_unit_test_ops):
    # print(op)

all_gather_dispatcher = plugin_unit_test_ops.all_gather_dispatcher
csr_conversion_distributed = plugin_unit_test_ops.csr_conversion_distributed
reduce_scatter_dispatcher = plugin_unit_test_ops.reduce_scatter_dispatcher