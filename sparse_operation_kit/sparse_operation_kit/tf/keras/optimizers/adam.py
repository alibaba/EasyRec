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

from .common import _deduplicate_indexed_slices
import tensorflow.keras.optimizers as keras_optimizers

class Adam(keras_optimizers.Adam):
    """
    Abbreviated as ``sok.tf.keras.optimizers.Adam``.

    The ``unique`` and ``unsorted_segment_sum`` are replaced with GPU
    implementations. 

    When TF version <= 2.4, this optimizer can be used to obtain performance
    gain, while TF version >= 2.5, its performance should be identical to 
    ``tf.keras.optimizers.Adam``.

    All the arguments are identical to ``tf.keras.optimizers.Adam``, please
    refer to https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adam 
    for documentation.
    """
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                               **kwargs):
        summed_grad, unique_indices = _deduplicate_indexed_slices(
            values=grad, indices=indices)
        return self._resource_apply_sparse(summed_grad, handle, unique_indices,
                                           **kwargs)