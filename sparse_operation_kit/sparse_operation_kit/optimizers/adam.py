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

from sparse_operation_kit import kit_lib
from sparse_operation_kit.core import EmbeddingVariable
from sparse_operation_kit.core.graph_keys import GraphKeys as sok_GraphKeys
from sparse_operation_kit.optimizers import optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.keras import backend_config

class Adam(optimizer.Optimizer):
    def __init__(self,
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                name='Plugin_Adam',
                **kwargs):
        super(Adam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self._beta1 = beta_1
        self._beta2 = beta_2

        self._optimizer_handler = kit_lib.create_global_adam_optimizer(beta1=self._beta1,
                                                               beta2=self._beta2,
                                                               epsilon=self.epsilon)

        if not kit_lib.in_tensorflow2():
            collections = [sok_GraphKeys.SparseOperationKitOptimizer]
            ops.add_to_collections(collections, self)

            _initializer_op = kit_lib.optimizer_init(self._optimizer_handler)
            self._initializer_op = control_flow_ops.group(_initializer_op)

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
        if not isinstance(var, EmbeddingVariable):
            raise ValueError("This optimizer can only be used to EmbeddingVariable.")

        var_device, var_dtype = var.device, var.dtype.base_dtype
        learning_rate = kwargs["apply_state"].get((var_device, var_dtype))["lr_t"]

        current_iteration = array_ops.identity(self.iterations)

        update_op = kit_lib.custom_optimizer_apply_gradients(emb_var_handle=var.m_handle, 
                                                    grad=grad,
                                                    local_indices=indices, 
                                                    learning_rate=learning_rate,
                                                    current_step=current_iteration)
        return control_flow_ops.group((update_op))


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
        config = super(Adam, self).get_config()
        config.update()
        raise NotImplementedError("get_config not implemented")

