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

"""
This folder contains TF ops that might not be in current TF version.

For example, the tf.unique has GPU implementation in TF 2.5, but it
is not so in TF 1.15. Therefore in this folder, we place it here.

And the module chains originated from TF is honored. For example, 
sok.tf.unique is equal to tf.unique in TF 2.5.
"""
import sparse_operation_kit.tf.keras

from sparse_operation_kit.operations import unique
from sparse_operation_kit.operations import unsorted_segment_sum

__all__ = [item for item in dir() if not item.startswith("__")]