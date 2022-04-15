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
        os.path.dirname(os.path.abspath(__file__)), r"../../../")))
import sparse_operation_kit as sok
import tensorflow as tf

class SOKDemo(tf.keras.models.Model):
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 slot_num,
                 max_nnz,
                 embedding_vec_size, 
                 use_hashtable=True,
                 key_dtype=None,
                 embedding_initializer=None,
                 **kwargs):
        super(SOKDemo, self).__init__(**kwargs)

        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.embedding_vec_size = embedding_vec_size

        self.embedding_layer = sok.DistributedEmbedding(combiner=self.combiner,
                                                        max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                        embedding_vec_size=self.embedding_vec_size,
                                                        slot_num=self.slot_num,
                                                        max_nnz=self.max_nnz,
                                                        use_hashtable=use_hashtable,
                                                        key_dtype=key_dtype,
                                                        embedding_initializer=embedding_initializer)

        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize, slot_num, embedding_vec_size]
        embedding_vector = self.embedding_layer(inputs, training=training)
        # [batchsize, slot_num * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.embedding_vec_size])
        # [batchsize, 1]
        logit = self.dense_layer(embedding_vector)
        return logit, embedding_vector

class TFSparseEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 init_tensors,
                 combiner,
                 **unused):
        super(TFSparseEmbedding, self).__init__()

        self._init_tensors = init_tensors
        self._combiner = combiner

    def build(self, input_shape):
        self._params = tf.Variable(initial_value=tf.concat(self._init_tensors, axis=0),
                                   dtype=self._dtype_policy.variable_dtype)

    def call(self, inputs):
        outputs = tf.nn.embedding_lookup_sparse(params=self._params, sp_ids=inputs, 
                                             sp_weights=None, combiner=self._combiner)
        return tf.cast(outputs, self._dtype_policy.compute_dtype)

def create_SOKSparseDemo_model(combiner, max_vocabulary_size_per_gpu,
                               slot_num, max_nnz, embedding_vec_size, use_hashtable=True):
    input_tensor = tf.keras.Input(type_spec=tf.SparseTensorSpec(shape=(slot_num, max_nnz), 
                                  dtype=tf.int64))
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
    model.embedding_layer = embedding_layer
    return model


class TfDemo(tf.keras.models.Model):
    def __init__(self, 
                 init_tensors, 
                 combiner, 
                 global_batch_size,
                 slot_num, 
                 embedding_vec_size,
                 **kwargs):
        super(TfDemo, self).__init__(**kwargs)
        self.combiner = combiner
        self.global_batch_size = global_batch_size
        self.slot_num = slot_num
        self.embedding_vec_size = embedding_vec_size

        self.init_tensors = init_tensors
        self.embedding_layer = TFSparseEmbedding(init_tensors=init_tensors, 
                                                 combiner=combiner)

        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    @property
    def params(self):
        return self.embedding_layer._params

    def call(self, inputs, training=True):
        # [batchsize * slot_num, embedding_vec_size]
        embedding_vector = self.embedding_layer(inputs)

        # [batchsize, slot_num * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, 
            shape=[self.global_batch_size, self.slot_num * self.embedding_vec_size])
        logit = self.dense_layer(embedding_vector)
        return logit, embedding_vector