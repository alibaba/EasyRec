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

import tensorflow as tf
from tensorflow.python.ops import collective_ops
from tensorflow.python.framework import ops

class TFDistributedEmbedding(tf.keras.layers.Layer):
    """
    This Embedding layer will distribute embedding parameters
    to multiple GPUs. It leverages tf.distribute.Strategy to
    do the communication, so that tf.distribute.Strategy must be 
    used. 

    Parameters
    ----------
    vocabulary_size: integer
            the first dimension of variable whose shape is 
            [vocabulary_size, embedding_vec_size].
    embedding_vec_size: integer
            the second dimension of variable whose shape is 
            [vocabulary_size, embedding_vec_size].
    initializer: string, numpy.array = 'GlorotNormal'
            When it's string, it specifies the initializer used to generate initial values.
            When it's numpy.array, its shape must be [vocabulary_size, embedding_vec_size],
            and will be used as the initial value.
    comm_options: tf.distribute.experimental.CommunicationOptions = None
            see TF's docs

    Examples
    --------
    .. code-block:: python

        strategy = ...

        with strategy.scope():
            embedding_layer = TFDistributedEmbedding(vocabulary_size, embedding_vec_size,
                                                     initializer)
            ...

        @tf.function
        def _train_step(inputs, labels):
            emb_vectors = embedding_layer(inputs)

        for i, (inputs, labels) in enumerate(dataset):
            strategy.run(_train_step, args=(inputs, labels))

    Notes
    -----
    Currently, the variables created by this class can not be correctly saved to files.
    """
    def __init__(self, 
                 vocabulary_size,
                 embedding_vec_size,
                 initializer="GlorotNormal",
                 comm_options=None,
                 **kwargs):
        super(TFDistributedEmbedding, self).__init__(**kwargs)

        self._vocabulary_size = vocabulary_size
        self._embedding_vec_size = embedding_vec_size
        self._uid = ops.uid()

        if isinstance(initializer, str):
            self._initial_value = tf.keras.initializers.get(initializer)(
                    shape=(self._vocabulary_size, self._embedding_vec_size))
        else:
            if initializer.shape != (self._vocabulary_size, self._embedding_vec_size):
                raise ValueError("The shape of initializer must be [vocabulary_size, embedding_vec_size.]")
            self._initial_value = initializer

        self._comm_options = comm_options

        self._embedding_weights = tf.Variable(
                initial_value=self._initial_value,
                dtype=tf.float32,
                name="EmbeddingWeights")

        if not tf.distribute.has_strategy():
            raise RuntimeError("This layer must be created under tf.distribute.Strategy.Scope().")
        # strategy = tf.distribute.get_strategy()
        # strategy.run(self.broadcast_variables)

    @property
    def embedding_weights(self):
        return self._embedding_weights
                
    @tf.function
    def broadcast_variables(self):
        replica_ctx = tf.distribute.get_replica_context()
        g_replica_id = replica_ctx.replica_id_in_sync_group
        if replica_ctx.num_replicas_in_sync == 1:
            return

        variable = tf.identity(self._embedding_weights)
        if 0 == g_replica_id:
            values = collective_ops.broadcast_send(variable,
                                                   variable.shape,
                                                   variable.dtype,
                                                   group_size=replica_ctx.num_replicas_in_sync,
                                                   group_key=2,
                                                   instance_key=2 + self._uid,
                                                   timeout=5)
        else:
            values = collective_ops.broadcast_recv(variable.shape,
                                                   variable.dtype,
                                                   group_size=replica_ctx.num_replicas_in_sync,
                                                   group_key=2,
                                                   instance_key=2 + self._uid,
                                                   timeout=5)
        self._embedding_weights.assign(values)

    def _condition(self, gathered_inputs, replica_ctx):
        global_replica_id = replica_ctx.replica_id_in_sync_group
        global_replica_id = tf.cast(global_replica_id, gathered_inputs.dtype)
        num_devices = replica_ctx.num_replicas_in_sync

        condition = (gathered_inputs % num_devices == global_replica_id)
        return condition

    def call(self, inputs):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        inputs: inputs: tf.Tensor
                keys are stored in tf.Tensor with dtype tf.int32 or tf.int64

        Returns
        -------
        replica_output: tf.Tensor
                embedding vectors on each replica, with dtype tf.float32
        """
        if tf.distribute.in_cross_replica_context():
            raise RuntimeError("The forward propagation of TFDistributedEmbedding "
                               "cannot be called in cross_replica_context.")
        replica_ctx = tf.distribute.get_replica_context()
        global_replica_id = replica_ctx.replica_id_in_sync_group

        inputs_shape = tf.shape(inputs)
        replica_size = tf.size(inputs)
        replica_inputs = tf.reshape(inputs, [replica_size])
        replica_inputs = tf.identity(replica_inputs)
        # all-gather for each replica along batch dim
        gathered_inputs = replica_ctx.all_gather(value=replica_inputs, axis=0,
                                                 options=self._comm_options)

        # select inputs for each replica
        condition = self._condition(gathered_inputs, replica_ctx)
        replica_indices = tf.where(condition)
        replica_selected_inputs = tf.gather_nd(gathered_inputs, replica_indices)

        # embedding lookup 
        replica_vectors = tf.nn.embedding_lookup(params=self._embedding_weights, 
                                                 ids=replica_selected_inputs)

        # all-gather embedding vectors for each replica
        gathered_vectors = replica_ctx.all_gather(value=replica_vectors, axis=0,
                                                  options=self._comm_options)
        gathered_indices = replica_ctx.all_gather(value=replica_indices, axis=0,
                                                  options=self._comm_options)
        gathered_indices = tf.squeeze(gathered_indices)

        # reorder embedding vectors
        sorted_gathered_indices = tf.argsort(gathered_indices)
        gathered_inputs_size = replica_ctx.all_gather(value=tf.expand_dims(replica_size, axis=0), 
                                                      axis=0, options=self._comm_options)
        if tf.rank(gathered_inputs_size) == 0: 
            gathered_inputs_size = tf.expand_dims(gathered_inputs_size, axis=0)
        begin = tf.math.reduce_sum(gathered_inputs_size[:global_replica_id]) if global_replica_id > 0 else 0
        begin = tf.expand_dims(begin, axis=0)
        size = tf.slice(gathered_inputs_size, begin=[global_replica_id], size=[1])
        replica_output_indices = tf.slice(sorted_gathered_indices, 
                                          begin=begin,
                                          size=size)

        # select replica's vectors 
        replica_output = tf.gather(gathered_vectors, replica_output_indices)
        output_shape = inputs.get_shape().concatenate(self._embedding_vec_size)
        replica_output = tf.reshape(replica_output, output_shape)
        return replica_output


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    input_data = tf.constant([i for i in range(40)])
    input_data = tf.reshape(input_data, [8, 5])

    initial_value = tf.constant([i for i in range(40 * 4)], dtype=tf.float32)
    initial_value = tf.reshape(initial_value, shape=(40, 4))

    global_batch_size = 4

    strategy = tf.distribute.MirroredStrategy()

    def _dataset_fn(input_context):
        replica_bs = input_context.get_per_replica_batch_size(global_batch_size)

        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(replica_bs)
        dataset = dataset.repeat(2)
        dataset = dataset.shard(input_context.num_input_pipelines, 
                                input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)

    with strategy.scope():
        embedding_layer = TFDistributedEmbedding(vocabulary_size=40,
                                                 embedding_vec_size=4,
                                                 initializer=initial_value)
    @tf.function
    def _step(inputs):
        with tf.GradientTape() as tape:
            outputs = embedding_layer(inputs)
        grads = tape.gradient(outputs, embedding_layer._embedding_weights)
        return outputs, grads

    for step, inputs in enumerate(dataset):
        outputs = strategy.run(_step, args=(inputs,))
        print(f"Iteration: {step}")
        for out in outputs:
            print(out)