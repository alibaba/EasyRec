#
# Copyright (c) 2021, NVIDIA CORPORATION.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit.core import EmbeddingVariable
from tensorflow.python.distribute.values import MirroredVariable
from tensorflow.python.distribute.values import DistributedVariable

def split_embedding_variable_from_others(variables):
    """
    This function is used to split embedding variables from other variables.

    Abbreviated as ``sok.split_embedding_variable_from_others(variables)``.

    Embedding variables are automatically created along with embedding layers. 
    Since the aggregation for embedding variables is different from other variables, 
    we need to split embedding variable and other variables so that optimizer can 
    process those variables in different way.

    Parameters
    ----------
    variables: list, tuple
            a list or tuple of trainable *tf.Variable*.

    Returns
    -------
    embedding_variables: tuple
        all embedding variables in the input variable-list.
    other_variables: tuple
        all normal variables in the input variable-list.

    Example
    -------
    .. code-block:: python

        class Model(tf.keras.models.Model):
            def __init__(self, *args, **kwargs):
                super(Model, self).__init__(*args, **kwargs)

                self.embedding_layer = sok.DistributedEmbedding(...)
                self.dense_layer = tf.keras.layers.Dense(units=1, ...)

            def call(self, inputs, training=True):
                vectors = self.embedding_layer(inputs, training)
                out = self.dense_layer(vectors)
                return out

        with strategy.scope():
            model = Model()

        loss_fn = ...
            
        @tf.function
        def _train_step(inputs, labels):
            with tf.GradientTape() as tape:
                out = model(inputs)
                loss = loss_fn(out, labels)
            emb_vars, other_vars = sok.split_embedding_variable_from_others(model.trainable_variables)
            ...

        for step, (inputs, labels) in enumerate(dataset):
            strategy.run(_train_step, args=(inputs, labels))
            ...
    """
    if not (isinstance(variables, list) or isinstance(variables, tuple)):
        raise ValueError("Variables must be list or tuple. But got ", type(variables))

    embedding_variables = list()
    other_variables = list()

    for variable in variables:
        if (isinstance(variable, DistributedVariable) and 
            not isinstance(variable, MirroredVariable)):
            if isinstance(variable.values[0], EmbeddingVariable):
                embedding_variables.append(variable)
            else:
                other_variables.append(variable)
        elif isinstance(variable, EmbeddingVariable):
            # horovod branch
            embedding_variables.append(variable)
        else:
            other_variables.append(variable)

    return embedding_variables, other_variables
