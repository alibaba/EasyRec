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
from tensorflow.contrib import opt
from tensorflow.python.framework import ops

class LazyAdamOptimizer(opt.LazyAdamOptimizer):
    """
    Abbreviated as ``sok.tf.keras.optimizers.LazyAdamOptimizer``.

    The ``unique`` and ``unsorted_segment_sum`` are replaced with GPU
    implementations. 

    It is only available in TensorFlow 1.15. Although the name is `LazyAdam`,
    it actually updates variables locally.
    """
    def __init__(self, *args, **kwargs):
        super(LazyAdamOptimizer, self).__init__(*args, **kwargs)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        """Add ops to apply sparse gradients to `handle`, with repeated indices.

        Optimizers which override this method must deal with repeated indices. See
        the docstring of `_apply_sparse_duplicate_indices` for details. By default
        the correct behavior, to sum non-unique indices and their associated
        gradients, is enforced by first pre-processing `grad` and `indices` and
        passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
        with duplicate indices may instead override this method to avoid the
        overhead of summing.

        Args:
            grad: a `Tensor` representing the gradient for the affected indices.
            handle: a `Tensor` of dtype `resource` which points to the variable
                to be updated.
            indices: a `Tensor` of integral type representing the indices for
                which the gradient is nonzero. Indices may be repeated.

        Returns:
            An `Operation` which updates the value of the variable.
        """
        summed_grad, unique_indices = _deduplicate_indexed_slices(
            values=grad, indices=indices)
        return self._resource_apply_sparse(summed_grad, handle, unique_indices)

    def _apply_sparse_duplicate_indices(self, grad, var):
        """Add ops to apply sparse gradients to `var`, with repeated sparse indices.

        Optimizers which override this method must deal with IndexedSlices objects
        such as the following:

            IndexedSlicesValue(values=[1, 1], indices=[0, 0], dense_shape=[1])

        The correct interpretation is:

            IndexedSlicesValue(values=[2], indices=[0], dense_shape=[1])

        Many optimizers deal incorrectly with repeated indices when updating based
        on sparse gradients (e.g. summing squares rather than squaring the sum, or
        applying momentum terms multiple times). Adding first is always the correct
        behavior, so this is enforced here by reconstructing the IndexedSlices to
        have only unique indices, then calling _apply_sparse.

        Optimizers which deal correctly with repeated indices may instead override
        this method to avoid the overhead of summing indices.

        Args:
            grad: `IndexedSlices`.
            var: A `Variable` object.

        Returns:
            An `Operation`.
        """
        summed_values, unique_indices = _deduplicate_indexed_slices(
            values=grad.values, indices=grad.indices)
        gradient_no_duplicate_indices = ops.IndexedSlices(
            indices=unique_indices,
            values=summed_values,
            dense_shape=grad.dense_shape)
        return self._apply_sparse(gradient_no_duplicate_indices, var)
