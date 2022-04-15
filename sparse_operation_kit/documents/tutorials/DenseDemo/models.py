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

import tensorflow as tf
import sys
sys.path.append("../../../")
import sparse_operation_kit as sok
from sparse_operation_kit.embeddings.tf_distributed_embedding import TFDistributedEmbedding

from tensorflow.python.framework import ops

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

        return embedding


class SOKDenseDemo(tf.keras.models.Model):
    def __init__(self, 
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size,
                 slot_num, 
                 nnz_per_slot,
                 num_dense_layers,
                 **kwargs):
        super(SOKDenseDemo, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot
        self.num_dense_layers = num_dense_layers
        self.embedding_vec_size = embedding_vec_size

        self.embedding_layer = sok.All2AllDenseEmbedding(max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                         embedding_vec_size=self.embedding_vec_size,
                                                         slot_num=self.slot_num,
                                                         nnz_per_slot=self.nnz_per_slot)
        
        self.dense_layers = []
        for _ in range(self.num_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu")
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize, slot_num, nnz_per_slot, embedding_vec_size]
        embedding_vector = self.embedding_layer(inputs, training=training)
        # [batchsize, slot_num * nnz_per_slot * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.nnz_per_slot * self.embedding_vec_size])
        
        hidden = embedding_vector
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        # [batchsize, 1]
        logit = self.out_layer(hidden)
        return logit


class TfDenseDemo(tf.keras.models.Model):
    def __init__(self,
                 global_batch_size,
                 vocabulary_size,
                 slot_num, 
                 nnz_per_slot,
                 num_dense_layers,
                 embedding_vec_size,
                 **kwargs):
        super(TfDenseDemo, self).__init__(**kwargs)
        self.global_batch_size = global_batch_size
        self.vocabulary_size = vocabulary_size
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot
        self.num_dense_layers = num_dense_layers
        self.embedding_vec_size = embedding_vec_size

        self.embedding_layer = TFDistributedEmbedding(vocabulary_size=self.vocabulary_size,
                                                      embedding_vec_size=self.embedding_vec_size)

        self.dense_layers = []
        for _ in range(self.num_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation='relu')
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize * slot_num * nnz_per_slot, embedding_vec_size]
        embedding_vector = self.embedding_layer(inputs=inputs, training=training)

        # [batchsize, slot_num * nnz_per_slot * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, 
            shape=[-1, self.slot_num * self.nnz_per_slot * self.embedding_vec_size])
        
        hidden = embedding_vector
        for layer in self.dense_layers:
            hidden = layer(hidden)

        # [batchsize, 1]
        logit = self.out_layer(hidden)
        return logit


class SOKDenseModel(tf.keras.models.Model):
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size_list,
                 slot_num_list,
                 nnz_per_slot_list,
                 num_dense_layers,
                 dynamic_input = False,
                 use_hashtable = True,
                 **kwargs):
        super(SOKDenseModel, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size_list = embedding_vec_size_list
        self.slot_num_list = slot_num_list
        self.nnz_per_slot_list = nnz_per_slot_list
        self.num_dense_layers = num_dense_layers
        self.dynamic_input = dynamic_input

        if (len(slot_num_list) != len(nnz_per_slot_list) or 
            len(slot_num_list) != len(embedding_vec_size_list)):
            raise ValueError("The length of embedding_vec_size_list, slot_num_list"+\
                             " and nnz_per_slot_list must be equal.")

        self.embedding_num = len(self.embedding_vec_size_list)
        self.slot_num_prefix_sum = [0 for _ in range(self.embedding_num + 1)]
        for i in range(1, self.embedding_num + 1):
            self.slot_num_prefix_sum[i] = self.slot_num_prefix_sum[i-1] + self.slot_num_list[i-1]

        self.embedding_layers = list()
        for i in range(self.embedding_num):
            self.embedding_layer = sok.All2AllDenseEmbedding(max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                             embedding_vec_size=self.embedding_vec_size_list[i],
                                                             slot_num=self.slot_num_list[i],
                                                             nnz_per_slot=self.nnz_per_slot_list[i],
                                                             dynamic_input=self.dynamic_input,
                                                             use_hashtable=use_hashtable)
            self.embedding_layers.append(self.embedding_layer)

        self.dense_layers = list()
        for _ in range(self.num_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu",
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")

    def do_lookup(self, embedding_layer, inputs, training):
        if self.dynamic_input:
            _unique_inputs, _unique_index = tf.unique(x=tf.reshape(inputs, shape=[-1]))
            _emb_vector = embedding_layer(_unique_inputs, training=training)
            embedding_vector = tf.gather(_emb_vector, _unique_index)
        else:
            embedding_vector = embedding_layer(inputs, training=training)
        return embedding_vector

    def call(self, inputs, training=True):
        """
        The inputs has shape: [batchsize, slot_num, nnz_per_slot]
        split it along slot-num axis into self.embedding_num shards.
        """
        vectors = list()

        embedding_vector = self.do_lookup(self.embedding_layers[0], 
                                          inputs[:,self.slot_num_prefix_sum[0]:self.slot_num_prefix_sum[0+1],:],
                                          training=training)

        # [batchsize, slot_num * nnz_per_slot * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num_list[0] * self.nnz_per_slot_list[0] * self.embedding_vec_size_list[0]])
        vectors.append(embedding_vector)

        for i in range(1, self.embedding_num):
            with tf.control_dependencies([embedding_vector]):
                embedding_vector = self.do_lookup(self.embedding_layers[i],
                                                 inputs[:,self.slot_num_prefix_sum[i]:self.slot_num_prefix_sum[i+1],:],
                                                 training=training)

                # [batchsize, slot_num * nnz_per_slot * embedding_vec_size]
                embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num_list[i] * self.nnz_per_slot_list[i] * self.embedding_vec_size_list[i]])
                vectors.append(embedding_vector)

        all_vectors = tf.concat(values=vectors, axis=1)

        hidden = all_vectors
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit, all_vectors


class TFDenseModel(tf.keras.models.Model):
    def __init__(self, 
                 vocabulary_size,
                 embedding_vec_size_list,
                 slot_num_list,
                 nnz_per_slot_list,
                 num_dense_layers,
                 **kwargs):
        super(TFDenseModel, self).__init__(**kwargs)

        self.vocabulary_size = vocabulary_size
        self.embedding_vec_size_list = embedding_vec_size_list
        self.slot_num_list = slot_num_list
        self.nnz_per_slot_list = nnz_per_slot_list
        self.num_dense_layers = num_dense_layers

        if (len(slot_num_list) != len(nnz_per_slot_list) or 
            len(slot_num_list) != len(embedding_vec_size_list)):
            raise ValueError("The length of embedding_vec_size_list, slot_num_list" +\
                             " and nnz_per_slot_list must be equal.")

        self.embedding_num = len(self.embedding_vec_size_list) 
        self.slot_num_prefix_sum = [0 for _ in range(self.embedding_num + 1)]
        for i in range(1, self.embedding_num + 1):
            self.slot_num_prefix_sum[i] = self.slot_num_prefix_sum[i-1] + self.slot_num_list[i-1]

        self.embedding_params = list()
        for i in range(self.embedding_num):
            self.param = self.add_weight(shape=(self.vocabulary_size, self.embedding_vec_size_list[i]),
                                         dtype=tf.float32, name="embedding_table_"+str(i),
                                         initializer="glorot_normal")
            self.embedding_params.append(self.param)

        self.dense_layers = list()
        for _ in range(self.num_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu",
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")
            self.dense_layers.append(self.layer)
        
        self.out_layer = tf.keras.layers.Dense(units=1, activation=None,
                                               kernel_initializer="ones",
                                               bias_initializer="zeros")

    def call(self, inputs, training=True):
        vectors = list()

        embedding_vector = tf.nn.embedding_lookup(params=self.embedding_params[0],
                                                  ids=inputs[:,self.slot_num_prefix_sum[0]:self.slot_num_prefix_sum[0+1],:])
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num_list[0] * self.nnz_per_slot_list[0] * self.embedding_vec_size_list[0]])
        vectors.append(embedding_vector)

        for i in range(1, self.embedding_num):
            with tf.control_dependencies([embedding_vector]):
                embedding_vector = tf.nn.embedding_lookup(params=self.embedding_params[i],
                                        ids=inputs[:,self.slot_num_prefix_sum[i]:self.slot_num_prefix_sum[i+1],:])
                embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num_list[i] * self.nnz_per_slot_list[i] * self.embedding_vec_size_list[i]])
                vectors.append(embedding_vector)

        all_vectors = tf.concat(values=vectors, axis=1)

        hidden = all_vectors
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit, all_vectors
