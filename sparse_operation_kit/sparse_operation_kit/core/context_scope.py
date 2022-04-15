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

from sparse_operation_kit.core import EmbeddingVariable
from sparse_operation_kit.optimizers.utils import split_embedding_variable_from_others
from tensorflow.python.distribute.values import DistributedVariable, MirroredVariable

class OptimizerScope(object):
    """
    The context manager used along with TensorFlow optimizers. 
    It is only needed when TensorFlow native optimizers is used.

    Abbreviated as ``sok.OptimizerScope(variables)``.

    Parameters
    ----------
    trainable_variables: list, tuple
            a list or tuple of trainable *tf.Variable*.

    Returns
    -------
    context_manager: context_manager
            used to switch handles for embedding variables.

    Example
    -------
    .. code-block:: python

        with strategy.scope():
            model = ...

            emb_opt = tf.keras.optimizers.Adam(...)
            other_opt = tf.keras.optimizers.Adam(...)

        @tf.function
        def _train_step(inputs, labels):
            with tf.GradientTape() as tape:
                logits = model(inputs)
                loss = loss_fn(logits, labels)

            emb_vars, other_vars = sok.split_embedding_variable_from_others(model.trainable_variables)
            emb_grads, other_grads = tape.gradient(loss, [emb_vars, other_vars])

            with sok.OptimizerScope(emb_vars):
                emb_opt.apply_gradients(zip(emb_grads, emb_vars),
                                        experimental_aggregate_gradients=False)

            dense_opt.apply_gradients(zip(other_grads, other_vars))

    Notes
    -----
    This context manager may not be used in next release.
    """
    def __init__(self, trainable_variables):
        if not (isinstance(trainable_variables, list) or isinstance(trainable_variables, tuple)):
            raise RuntimeError("trainable_variables must be a list or tuple.")
        self._trainable_variables = trainable_variables

        self._embedding_variables, _ = split_embedding_variable_from_others(self._trainable_variables)

    def __enter__(self):
        self.touched_variables = list()
        
        for variable in self._embedding_variables:
            if isinstance(variable, EmbeddingVariable):
                # When using horovod, type(variable) is EmbeddingVariable
                variable._handle = variable.tf_handle
                self.touched_variables.append(variable)
            else:
                # When using tf.strategy, type(variable) is DistributedVariable
                for sub_variable in variable.values:
                    sub_variable._handle = sub_variable.tf_handle
                    self.touched_variables.append(sub_variable)

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        This scope does not process exception.
        """
        for variable in self.touched_variables:
            variable._handle = variable.m_handle
        return False