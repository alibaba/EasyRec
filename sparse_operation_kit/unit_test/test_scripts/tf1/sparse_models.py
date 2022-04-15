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

import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../")))
import sparse_operation_kit as sok
import tensorflow as tf

class SOKDemo(tf.keras.models.Model):
    def __init__(self, 
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size,
                 combiner,
                 slot_num, 
                 max_nnz,
                 use_hashtable=True,
                 num_of_dense_layers=5,
                 key_dtype=None,
                 embedding_initializer=None,
                 **unused):
        super(SOKDemo, self).__init__()

        self._max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self._embedding_vec_size = embedding_vec_size
        self._combiner = combiner
        self._slot_num = slot_num
        self._max_nnz = max_nnz
        self._use_hashtable = use_hashtable
        self._num_of_dense_layers = num_of_dense_layers
        self._key_dtype = key_dtype

        if (isinstance(self._embedding_vec_size, list) or
            isinstance(self._embedding_vec_size, tuple)):
            if len(self._embedding_vec_size) != len(self._slot_num):
                raise ValueError("The length of embedding_vec_size must be equal to "
                                 "that of slot_num")

        self._embedding_num = len(self._embedding_vec_size)
        self._slot_num_prefix_num = [0 for _ in range(self._embedding_num + 1)]
        for i in range(1, self._embedding_num + 1):
            self._slot_num_prefix_num[i] = self._slot_num_prefix_num[i-1] + self._slot_num[i-1]
        
        self.embedding_layers = list()
        for i in range(self._embedding_num):
            embedding_layer = sok.DistributedEmbedding(max_vocabulary_size_per_gpu=self._max_vocabulary_size_per_gpu,
                                                       embedding_vec_size=self._embedding_vec_size[i],
                                                       combiner=self._combiner,
                                                       slot_num=self._slot_num[i],
                                                       max_nnz=self._max_nnz,
                                                       use_hashtable=self._use_hashtable,
                                                       key_dtype=key_dtype,
                                                       embedding_initializer=embedding_initializer)
            self.embedding_layers.append(embedding_layer)

        self.dense_layers = list()
        for _ in range(self._num_of_dense_layers):
            layer = tf.keras.layers.Dense(units=1024, activation="relu",
                                          kernel_initializer="ones",
                                          bias_initializer="zeros")
            self.dense_layers.append(layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")

    def call(self, inputs, training=True):
        vectors = list()

        inputs = tf.sparse.reshape(inputs, [-1, sum(self._slot_num), self._max_nnz])
        if self._key_dtype == 'uint32':
            inputs = tf.sparse.SparseTensor(indices=inputs.indices,
                                            values=tf.cast(inputs.values, dtype=tf.int32),
                                            dense_shape=inputs.dense_shape)

        for i, embedding_layer in enumerate(self.embedding_layers):
            control_inputs = [vectors[-1]] if vectors else None
            with tf.control_dependencies(control_inputs):
                _input = tf.sparse.slice(inputs, [0, self._slot_num_prefix_num[i], 0],
                                        [tf.shape(inputs)[0], self._slot_num[i], tf.shape(inputs)[-1]])
                _input = tf.sparse.reshape(_input, [-1, self._max_nnz])
                if self._key_dtype == "uint32":
                    _input = tf.sparse.SparseTensor(indices=_input.indices,
                                                    values=tf.cast(_input.values, dtype=tf.uint32),
                                                    dense_shape=_input.dense_shape)
                embedding_vector = embedding_layer(_input, training)
                embedding_vector = tf.reshape(embedding_vector,
                                    shape=[-1, self._slot_num[i] * self._embedding_vec_size[i]])
                vectors.append(embedding_vector)

        all_vectors = tf.concat(values=vectors, axis=1)

        hidden = all_vectors
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        logit = self.out_layer(hidden)
        return logit, all_vectors

def create_SOKDemo(combiner,
                   max_vocabulary_size_per_gpu,
                   embedding_vec_size,
                   slot_num,
                   max_nnz,
                   use_hashtable=True):
    """
    Only 1 Embedding + 1 Dense
    """
    input_tensor = tf.keras.Input(shape=(max_nnz,), sparse=True, dtype=tf.int64)
    embedding_layer = sok.DistributedEmbedding(combiner=combiner,
                                               max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu,
                                               embedding_vec_size=embedding_vec_size,
                                               slot_num=slot_num,
                                               max_nnz=max_nnz,
                                               use_hashtable=use_hashtable)
    embedding = embedding_layer(input_tensor)
    embedding = tf.keras.layers.Reshape(target_shape=(slot_num * embedding_vec_size,))(embedding)

    logit = tf.keras.layers.Dense(units=1, activation=None,
                                  kernel_initializer="ones",
                                  bias_initializer="zeros")(embedding)
    model = tf.keras.Model(inputs=input_tensor, outputs=[logit, embedding])

    # set attr
    model.embedding_layers = [embedding_layer]
    return model


class TFDemo(tf.keras.models.Model):
    def __init__(self, 
                 vocabulary_size,
                 embedding_vec_size,
                 combiner,
                 slot_num, 
                 max_nnz,
                 use_hashtable=True,
                 num_of_dense_layers=5,
                 **unused):
        super(TFDemo, self).__init__()

        self._vocabulary_size = vocabulary_size
        self._embedding_vec_size = embedding_vec_size
        self._combiner = combiner
        self._slot_num = slot_num
        self._max_nnz = max_nnz
        self._use_hashtable = use_hashtable
        self._num_of_dense_layers = num_of_dense_layers

        if (isinstance(self._embedding_vec_size, list) or
            isinstance(self._embedding_vec_size, tuple)):
            if len(self._embedding_vec_size) != len(self._slot_num):
                raise ValueError("The length of embedding_vec_size must be equal to "
                                 "that of slot_num")

        self._embedding_num = len(self._embedding_vec_size)
        self._slot_num_prefix_num = [0 for _ in range(self._embedding_num + 1)]
        for i in range(1, self._embedding_num + 1):
            self._slot_num_prefix_num[i] = self._slot_num_prefix_num[i-1] + self._slot_num[i-1]
        
        initializer = tf.keras.initializers.get("uniform")
        self.embedding_weights = list()
        for i in range(self._embedding_num):
            weights = tf.Variable(initial_value=initializer(shape=(self._vocabulary_size, self._embedding_vec_size[i]),
                                                            dtype=tf.float32),
                                  name="tf_embeddings", use_resource=True)
            self.embedding_weights.append(weights)

        self.dense_layers = list()
        for _ in range(self._num_of_dense_layers):
            layer = tf.keras.layers.Dense(units=1024, activation="relu",
                                          kernel_initializer="ones",
                                          bias_initializer="zeros")
            self.dense_layers.append(layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")

    def call(self, inputs, training=True):
        vectors = list()

        inputs = tf.sparse.reshape(inputs, [-1, sum(self._slot_num), self._max_nnz])

        for i, embedding_weight in enumerate(self.embedding_weights):
            control_inputs = [vectors[-1]] if vectors else None
            with tf.control_dependencies(control_inputs):
                _input = tf.sparse.slice(inputs, [0, self._slot_num_prefix_num[i], 0],
                                        [tf.shape(inputs)[0], self._slot_num[i], tf.shape(inputs)[-1]])
                _input = tf.sparse.reshape(_input, [-1, self._max_nnz])
                embedding_vector = tf.nn.embedding_lookup_sparse(params=embedding_weight,
                                                                 sp_ids=_input,
                                                                 sp_weights=None,
                                                                 combiner=self._combiner)
                if (self._dtype_policy.compute_dtype and 
                    self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype):
                    embedding_vector = tf.cast(embedding_vector, self._dtype_policy.compute_dtype)

                embedding_vector = tf.reshape(embedding_vector,
                                    shape=[-1, self._slot_num[i] * self._embedding_vec_size[i]])
                vectors.append(embedding_vector)

        all_vectors = tf.concat(values=vectors, axis=1)

        hidden = all_vectors
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        logit = self.out_layer(hidden)
        return logit, all_vectors
