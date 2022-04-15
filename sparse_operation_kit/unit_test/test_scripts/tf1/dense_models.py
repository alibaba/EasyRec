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
                 slot_num,
                 nnz_per_slot,
                 use_hashtable=True,
                 dynamic_input=False,
                 num_of_dense_layers=5,
                 key_dtype=None,
                 embedding_initializer=None,
                 **unused):
        super(SOKDemo, self).__init__()

        self._max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self._embedding_vec_size = embedding_vec_size
        self._slot_num = slot_num
        self._nnz_per_slot = nnz_per_slot
        self._use_hashtable = use_hashtable
        self._dynamic_input = dynamic_input
        self._num_of_dense_layers = num_of_dense_layers

        if (isinstance(self._embedding_vec_size, list) or 
            isinstance(self._embedding_vec_size, tuple)):
            if len(self._embedding_vec_size) != len(self._slot_num):
                raise ValueError("The length of embedding_vec_size must be equal to that of "
                                 "slot_num")

        self._embedding_num = len(self._embedding_vec_size)
        self._slot_num_prefix_sum = [0 for _ in range(self._embedding_num + 1)]
        for i in range(1, self._embedding_num + 1):
            self._slot_num_prefix_sum[i] = self._slot_num_prefix_sum[i-1] + self._slot_num[i-1]
        
        self.embedding_layers = list()
        for i in range(self._embedding_num):
            embedding_layer = sok.All2AllDenseEmbedding(max_vocabulary_size_per_gpu=self._max_vocabulary_size_per_gpu,
                                                        embedding_vec_size=self._embedding_vec_size[i],
                                                        slot_num=self._slot_num[i],
                                                        nnz_per_slot=self._nnz_per_slot,
                                                        use_hashtable=self._use_hashtable,
                                                        dynamic_input=self._dynamic_input,
                                                        key_dtype=key_dtype,
                                                        embedding_initializer=embedding_initializer)
            self.embedding_layers.append(embedding_layer)

        self.dense_layers = list()
        for _ in range(self._num_of_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu", 
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                              kernel_initializer="ones",
                                              bias_initializer="zeros")

    def do_lookup(self, embedding_layer, inputs, training):
        if self._dynamic_input:
            inputs = tf.reshape(inputs, [-1])
            _unique_inputs, _unique_index = tf.unique(inputs)
            _unique_embedding_vector = embedding_layer(_unique_inputs, training=training)
            embedding_vector = tf.gather(_unique_embedding_vector, _unique_index)
        else:
            embedding_vector = embedding_layer(inputs, training=training)
        return embedding_vector

    def call(self, inputs, training=True):
        vectors = list()

        embedding_vector = self.do_lookup(self.embedding_layers[0],
                                          inputs[:,self._slot_num_prefix_sum[0]:self._slot_num_prefix_sum[0+1],:],
                                          training=training)
        embedding_vector = tf.reshape(embedding_vector, 
                    shape=[-1, self._slot_num[0] * self._nnz_per_slot * self._embedding_vec_size[0]])

        vectors.append(embedding_vector)

        for i in range(1, self._embedding_num):
            with tf.control_dependencies([embedding_vector]):
                embedding_vector = self.do_lookup(self.embedding_layers[i],
                                        inputs[:,self._slot_num_prefix_sum[i]:self._slot_num_prefix_sum[i+1],:],
                                        training=training)
                embedding_vector = tf.reshape(embedding_vector,
                        shape=[-1, self._slot_num[i] * self._nnz_per_slot * self._embedding_vec_size[i]])
                vectors.append(embedding_vector)
        
        all_vectors = tf.concat(values=vectors, axis=1)

        hidden = all_vectors
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit, all_vectors

def create_SOKDemo(max_vocabulary_size_per_gpu,
                   embedding_vec_size,
                   slot_num,
                   nnz_per_slot,
                   use_hashtable=True):
    """
    Only 1 Embedding + 1 Dense.
    """
    input_tensor = tf.keras.Input(shape=(slot_num, nnz_per_slot), dtype=tf.int64)
    embedding_layer = sok.All2AllDenseEmbedding(max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu,
                                                embedding_vec_size=embedding_vec_size,
                                                slot_num=slot_num,
                                                nnz_per_slot=nnz_per_slot,
                                                use_hashtable=use_hashtable)
    embedding = embedding_layer(input_tensor)
    embedding = tf.keras.layers.Reshape(target_shape=(slot_num * nnz_per_slot * embedding_vec_size,))(embedding)

    logit = tf.keras.layers.Dense(units=1, activation=None,
                                  kernel_initializer="ones",
                                  bias_initializer="zeros")(embedding)
    model = tf.keras.Model(inputs=input_tensor, outputs=[logit, embedding])

    # set attr
    model.embedding_layers = [embedding_layer]
    return model


class HashtableEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 max_vocabulary_size,
                 embedding_vec_size,
                 key_dtype=tf.int64,
                 value_dtype=tf.int64,
                 initializer='random_uniform',
                 serving_default_value=None 
                 ):
        super(HashtableEmbedding, self).__init__()
        self.max_vocabulary_size = max_vocabulary_size
        self.embedding_vec_size = embedding_vec_size
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.initializer = initializer
        self.serving_default_value = serving_default_value
        if (self.serving_default_value is not None
            and (not isinstance(self.serving_default_value, tf.Tensor)
            or not isinstance(self.serving_default_value, np.ndarray))):
                raise RuntimeError("serving_default_value must be None or tf.Tensor.")
        else:
            self.serving_default_value = tf.zeros(shape=[1, self.embedding_vec_size], dtype=tf.float32)

        self.minimum = -9223372036854775808
        self.maximum = 9223372036854775807

        self.default_value = tf.constant(self.minimum, dtype=self.value_dtype)

        if isinstance(self.initializer, str):
            self.initializer = tf.keras.initializers.get(self.initializer)
            initial_value = self.initializer(shape=[self.max_vocabulary_size, self.embedding_vec_size], dtype=tf.float32)
        elif isinstance(self.initializer, tf.keras.initializers.Initializer):
            initial_value = self.initializer(shape=[self.max_vocabulary_size, self.embedding_vec_size], dtype=tf.float32)
        elif isinstance(self.initializer, np.ndarray):
            initial_value = self.initializer
        else:
            raise RuntimeError("Not supported initializer.")

        self.hash_table = tf.lookup.experimental.DenseHashTable(
            key_dtype=self.key_dtype, value_dtype=self.value_dtype, default_value=self.default_value,
            empty_key=self.maximum, deleted_key=self.maximum - 1)
        self.counter = tf.Variable(initial_value=0, trainable=False, dtype=self.value_dtype, name="hashtable_counter")
        self.embedding_var = tf.Variable(initial_value=initial_value, dtype=tf.float32, name='embedding_variables')
        
        # used for inference, as the default embedding vector.
        self.default_embedding = tf.Variable(initial_value=tf.convert_to_tensor(self.serving_default_value, dtype=tf.float32),
                                                name='default_embedding_vector', trainable=False)

    def get_insert(self, flatten_ids, length):
        hash_ids = self.hash_table.lookup(flatten_ids)
        default_ids = tf.gather_nd(flatten_ids, tf.where(hash_ids == self.default_value))
        unique_default_ids, _ = tf.unique(default_ids)
        unique_default_ids_num = tf.size(unique_default_ids, out_type=self.value_dtype)
        if 0 != unique_default_ids_num:
            # TODO: check counter < max_vocabulary_size
            inserted_values = tf.range(start=self.counter, limit=self.counter + unique_default_ids_num, delta=1, dtype=self.value_dtype)
            self.counter.assign_add(unique_default_ids_num, read_value=False)
            self.hash_table.insert(unique_default_ids, inserted_values)
            hash_ids = self.hash_table.lookup(flatten_ids)

        return hash_ids

    def get(self, flatten_ids, length):
        hash_ids = self.hash_table.lookup(flatten_ids)
        hash_ids = tf.where(hash_ids == self.default_value, 
                            tf.constant(self.max_vocabulary_size, dtype=self.value_dtype),
                            hash_ids)
        return hash_ids

    @property
    def hashtable(self):
        return self.hash_table

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.int64), 
                                  tf.TensorSpec(dtype=tf.bool, shape=[])))
    def call(self, ids, training=True):
        flatten_ids = tf.reshape(ids, [-1])
        length = tf.size(flatten_ids)
        if training:
            hash_ids = self.get_insert(flatten_ids, length)
        else:
            hash_ids = self.get(flatten_ids, length)

        hash_ids = tf.reshape(hash_ids, tf.shape(ids))
        embedding = tf.nn.embedding_lookup([self.embedding_var, self.default_embedding], hash_ids)
        if (self._dtype_policy.compute_dtype and 
            self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype):
            embedding = tf.cast(embedding, self._dtype_policy.compute_dtype)

        return embedding


class Embedding(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Embedding, self).__init__()

        self._embedding = tf.keras.layers.Embedding(*args, **kwargs)

    @property
    def embeddings(self):
        return self._embedding.embeddings

    def call(self, *args, **kwargs):
        out = self._embedding(*args, **kwargs)
        if (self._dtype_policy.compute_dtype and 
            self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype):
            out = tf.cast(out, self._dtype_policy.compute_dtype)
        return out


class TFDemo(tf.keras.models.Model):
    def __init__(self,
                 vocabulary_size,
                 slot_num,
                 nnz_per_slot,
                 embedding_vec_size,
                 num_of_dense_layers=5,
                 use_hashtable=True,
                 dynamic_input=False,
                 **unused):
        super(TFDemo, self).__init__()

        self._vocabulary_size = vocabulary_size
        self._slot_num = slot_num
        self._nnz_per_slot = nnz_per_slot
        self._num_of_dense_layers = num_of_dense_layers
        self._embedding_vec_size = embedding_vec_size
        self._use_hashtable = use_hashtable
        self._dynamic_input = dynamic_input

        if (isinstance(self._embedding_vec_size, list) or 
            isinstance(self._embedding_vec_size, tuple)):
            if len(self._embedding_vec_size) != len(self._slot_num):
                raise ValueError("The length of embedding_vec_size must be equal to that of "
                                 "slot_num")
        self._embedding_num = len(self._embedding_vec_size)
        self._slot_num_prefix_sum = [0 for _ in range(self._embedding_num + 1)]
        for i in range(1, self._embedding_num + 1):
            self._slot_num_prefix_sum[i] = self._slot_num_prefix_sum[i-1] + self._slot_num[i-1]

        self.embedding_layers = list()
        for i in range(self._embedding_num):
            if self._use_hashtable:
                embedding_layer = HashtableEmbedding(max_vocabulary_size=self._vocabulary_size,
                                                    embedding_vec_size=self._embedding_vec_size[i])
            else:
                embedding_layer = Embedding(input_dim=self._vocabulary_size,
                                            output_dim=self._embedding_vec_size[i])
            self.embedding_layers.append(embedding_layer)

        self.dense_layers = list()
        for _ in range(self._num_of_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu", 
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                              kernel_initializer="ones",
                                              bias_initializer="zeros")

    def do_lookup(self, embedding_layer, inputs, training):
        if self._dynamic_input:
            inputs = tf.reshape(inputs, [-1])
            _unique_inputs, _unique_index = tf.unique(inputs)
            _unique_embedding_vector = embedding_layer(_unique_inputs, training=training)
            embedding_vector = tf.gather(_unique_embedding_vector, _unique_index)
        else:
            embedding_vector = embedding_layer(inputs, training=training)
        return embedding_vector

    def call(self, inputs, training=True):
        vectors = list()

        embedding_vector = self.do_lookup(self.embedding_layers[0],
                                          inputs[:,self._slot_num_prefix_sum[0]:self._slot_num_prefix_sum[0+1],:],
                                          training=training)
        embedding_vector = tf.reshape(embedding_vector, 
                    shape=[-1, self._slot_num[0] * self._nnz_per_slot * self._embedding_vec_size[0]])
        vectors.append(embedding_vector)

        for i in range(1, self._embedding_num):
            with tf.control_dependencies([embedding_vector]):
                embedding_vector = self.do_lookup(self.embedding_layers[i],
                                        inputs[:,self._slot_num_prefix_sum[i]:self._slot_num_prefix_sum[i+1],:],
                                        training=training)
                embedding_vector = tf.reshape(embedding_vector,
                        shape=[-1, self._slot_num[i] * self._nnz_per_slot * self._embedding_vec_size[i]])
                vectors.append(embedding_vector)
        
        all_vectors = tf.concat(values=vectors, axis=1)

        hidden = all_vectors
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit, all_vectors


        