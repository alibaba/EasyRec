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

class Initializer(object):
    def __init__(self, value):
        self._value = value

    def __call__(self, shape, dtype=None, **kwargs):
        return self._value

class SOKDenseDemo(tf.keras.models.Model):
    def __init__(self, 
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size,
                 slot_num, 
                 nnz_per_slot,
                 use_hashtable=True,
                 key_dtype=None,
                 embedding_initializer=None,
                 **kwargs):
        super(SOKDenseDemo, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot
        self.embedding_vec_size = embedding_vec_size

        self.embedding_layer = sok.All2AllDenseEmbedding(
                max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                embedding_vec_size=self.embedding_vec_size,
                slot_num=self.slot_num,
                nnz_per_slot=self.nnz_per_slot,
                use_hashtable=use_hashtable,
                key_dtype=key_dtype,
                embedding_initializer=embedding_initializer)
        
        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize, slot_num, nnz_per_slot, embedding_vec_size]
        embedding_vector = self.embedding_layer(inputs, training=training)
        # [batchsize, slot_num * nnz_per_slot * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.nnz_per_slot * self.embedding_vec_size])
        # [batchsize, 1]
        logit = self.dense_layer(embedding_vector)
        return logit, embedding_vector

def create_SOKDenseDemo_model(max_vocabulary_size_per_gpu, embedding_vec_size,
                              slot_num, nnz_per_slot, use_hashtable=True):
    input_tensor = tf.keras.Input(type_spec=tf.TensorSpec(shape=(None, slot_num, nnz_per_slot), 
                                  dtype=tf.int64))
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
    model.embedding_layer = embedding_layer
    return model


class TfDenseDemo(tf.keras.models.Model):
    def __init__(self,
                 init_tensors,
                 global_batch_size,
                 slot_num, 
                 nnz_per_slot,
                 embedding_vec_size,
                 **kwargs):
        super(TfDenseDemo, self).__init__(**kwargs)
        self.init_tensors = init_tensors
        self.global_batch_size = global_batch_size
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot
        self.embedding_vec_size = embedding_vec_size

        self.params = tf.Variable(initial_value=tf.concat(self.init_tensors, axis=0))

        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize * slot_num * nnz_per_slot, embedding_vec_size]
        embedding_vector = tf.nn.embedding_lookup(params=self.params,
                                                  ids=inputs)

        # [batchsize, slot_num * nnz_per_slot * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, 
            shape=[self.global_batch_size, self.slot_num * self.nnz_per_slot * self.embedding_vec_size])
        
        # [batchsize, 1]
        logit = self.dense_layer(embedding_vector)
        return logit, embedding_vector
