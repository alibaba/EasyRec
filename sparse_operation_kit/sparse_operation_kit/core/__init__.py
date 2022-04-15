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
from ._version import __version__

__all__ = ["initialize", "context_scope"]

from sparse_operation_kit.kit_lib import in_tensorflow2
if in_tensorflow2():
    from sparse_operation_kit.core.embedding_variable_v2 import EmbeddingVariable
else:
    from sparse_operation_kit.core.embedding_variable_v1 import EmbeddingVariable

from sparse_operation_kit.core.embedding_layer_handle import DenseEmbeddingLayerHandle
from sparse_operation_kit.core.embedding_layer_handle import SparseEmbeddingLayerHandle
