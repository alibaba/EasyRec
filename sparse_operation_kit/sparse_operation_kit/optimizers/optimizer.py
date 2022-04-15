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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_optimizer import optimizer_v2

class Optimizer(optimizer_v2.OptimizerV2):
    """
    The base class for sok's optimizer
    """
    # Subclasses should set this to True unless they override `apply_gradients`
    # with a version that does not have the `experimental_aggregate_gradients`
    # argument.  Older versions of Keras did not have this argument so custom
    # optimizers may have overridden `apply_gradients` without the
    # `experimental_aggregate_gradients` argument. Keras only passes
    # `experimental_aggregate_gradients` if this attribute is True.
    # Note: This attribute will likely be removed in an upcoming release.
    _HAS_AGGREGATE_GRAD = False

    def __init__(self, name, **unused):
        super(Optimizer, self).__init__(name)

    @property
    def initializer(self):
        return self._initializer_op

    def _resource_apply_dense(self, grad, var, apply_state):
        """
        update variable given gradient tensor is a dense `tf.Tensor`
        Args:
            grad: a `Tensor` representing the gradient.
            var: a `Tensor` of dtype `resource` which points to the variable to be
                updated.
            apply_state: A dict which is used across multiple apply calls.
        Returns:
            An `Operation` which updates the value of the variable.
        """
        raise NotImplementedError("_resource_apply_dense not implemented. "+\
                                  "please use other optimizers, such as "+\
                                  "tf.keras.optimiers.SGD, tf.keras.optimizers.Adam, etc.")

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        """
        update variable given gradient tensor is a
        sparse `tf.IndexedSlices`. The most common way for this to happen
        is if you are taking the gradient through a `tf.gather`.

        Similar to `_apply_sparse`, the `indices` argument to this method has been
        de-duplicated. Optimizers which deal correctly with non-unique indices may
        instead override `_resource_apply_sparse_duplicate_indices` to avoid this
        overhead.

        Args:
            grad: a `Tensor` representing the gradient for the affected indices.
            handle: a `Tensor` of dtype `resource` which points to the variable to be
                updated.
            indices: a `Tensor` of integral type representing the indices for which
                the gradient is nonzero. Indices are unique.
            apply_state: A dict which is used across multiple apply calls.
        Returns:
            An `Operation` which updates the value of the variable.
        """
        raise NotImplementedError("_resource_apply_sparse not implemented")

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
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
        handle: a `Tensor` of dtype `resource` which points to the variable to be
            updated.
        indices: a `Tensor` of integral type representing the indices for which
            the gradient is nonzero. Indices may be repeated.
        **kwargs: May optionally contain `apply_state`
            `apply_state` is a dict, which contains the map from (var_device, var_type) to 
                a dict and that dict contains "lr_t" which is learning_rate. In other words,
                apply_state might looks like: {(var_device, var_type): {"lr_t": learning_rate_tensor}}

        Returns:
        An `Operation` which updates the value of the variable.
        """
        raise NotImplementedError("_resource_apply_sparse_duplicate_indices not implemented")

    def _create_slots(self, var_list):
        """
        if your optimizer algorithm requires additional variables
        """
        pass

    def get_config(self):
        """
        serialization of the optimizer, include all hyper parameters

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Returns:
            Python dictionary.
        """
        raise NotImplementedError("get_config not implemented")