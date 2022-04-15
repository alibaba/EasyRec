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
from sparse_operation_kit.core import DenseEmbeddingLayerHandle
from sparse_operation_kit.embeddings import embedding_ops
import tensorflow as tf

class All2AllDenseEmbedding(tf.keras.layers.Layer):
    """
    Abbreviated as ``sok.All2AllDenseEmbedding(*args, **kwargs)``.

    This is a wrapper class for all2all dense embedding layer.
    It can be used to create a dense embedding layer which will distribute
    keys based on `gpu_id = key % gpu_num` to each GPU.

    Parameters
    ----------
    max_vocabulary_size_per_gpu: integer
            the first dimension of embedding variable whose shape is 
            [max_vocabulary_size_per_gpu, embedding_vec_size].
    embedding_vec_size: integer
            the second dimension of embedding variable whose shape is 
            [max_vocabulary_size_per_gpu, embedding_vec_size].
    slot_num: integer
            the number of feature-fileds which will be processed at the same time in
            each iteration, where all feature-fileds produce embedding vectors
            of the same dimension.
    nnz_per_slot: integer
            the number of valid keys in each slot. The number of valid keys in each slot 
            is the same.
    dynamic_input: boolean = False
            whether the inputs.shape is dynamic. For example, the inputs tensor is comming 
            from `tf.unique`. When `dynamic_input=True`, `unique->lookup->gather` pattern 
            can be used. By default, it is False, which means the inputs.size must be 
            `replica_batchsize * slot_num * nnz_per_slot`.
    use_hashtable: boolean = True
            whether using `Hashtable` in ``EmbeddingVariable``, if `True`,
            Hashtable will be created for dynamic insertion. Otherwise, the input keys
            will be used as the index for embedding vector looking-up, so that input keys
            must be in the range ``[0, max_vocabulary_size_per_gpu * gpu_num)``.
    key_dtype: tf.dtypes = tf.int64
            the data type of input keys. By default, it is `tf.int64`.
    embedding_initializer: string or an instance of `tf.keras.initializers.Initializer`
            the initializer used to generate initial value for embedding variable.
            By default, it will use `random_uniform` where ``minval=-0.05, maxval=0.05``.

    Examples
    --------
    .. code-block:: python

        initializer = tf.keras.initializers.RandomUniform() # or "random_uniform"

        emb_layer = sok.All2AllDenseEmbedding(max_vocabulary_size_per_gpu, 
                                              embedding_vec_size, 
                                              slot_num, nnz_per_slot,
                                              embedding_initializer=initializer)
        
        @tf.function
        def _train_step(inputs, labels):
            emb_vectors = emb_layer(inputs)
            ...
        
        for i, (inputs, labels) in enumerate(dataset):
            _train_step(inputs)
    """
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size, 
                 slot_num,
                 nnz_per_slot,
                 dynamic_input=False,
                 use_hashtable=True,
                 key_dtype=None,
                 embedding_initializer=None,
                 **kwargs):
        super(All2AllDenseEmbedding, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size = embedding_vec_size
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot
        self.dynamic_input = dynamic_input
        self.use_hashtable = use_hashtable

        if self._dtype_policy.variable_dtype is None:
            # in TF1 and policy is not set
            # therefore variable dtype and compute dtype should be fp32
            from tensorflow.python.keras.mixed_precision import experimental as mixed_precision
            self._dtype_policy = mixed_precision.Policy("float32")

        self.var = EmbeddingVariable.CreateInstances(
                        shape=[self.max_vocabulary_size_per_gpu, self.embedding_vec_size],
                        trainable=True,
                        use_hashtable=self.use_hashtable,
                        dtype=self._dtype_policy.variable_dtype,
                        key_dtype=key_dtype,
                        initializer=embedding_initializer)

        self.emb_layer = DenseEmbeddingLayerHandle(self.var,
                                                input_dispatcher="All2AllInput",
                                                embedding_lookuper="dense_gather",
                                                output_dispatcher="All2AllOutput",
                                                slot_num=self.slot_num,
                                                nnz_per_slot=self.nnz_per_slot,
                                                compute_dtype=self._dtype_policy.compute_dtype)

    @property
    def embedding_variable(self):
        return self.var

    # @tf.function
    def call(self, inputs, training=True):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        inputs: tf.Tensor
                keys are stored in tf.Tensor. It must be stored in row-major.
                If `dynamic_input = True`, then inputs.shape must be [None,], 
                otherwise, inputs.shape must be [batchsize, slot_num, nnz_per_slot]. 
        training: boolean
                whether training or not.

        Returns
        -------
        emb_vector: tf.float
                the embedding vectors for the input keys. When dynamic_input=False, 
                its shape is *[batchsize, slot_num, nnz_per_slot, embedding_vec_size]*.
                Otherwise, its shape is *[None, embedding_vec_size]*, where *None* equals
                to the size of inputs.
        """
        emb_vector = embedding_ops.embedding_lookup(embedding_variable=self.var, 
                                                    values=inputs,
                                                    training=training,
                                                    dynamic_input=self.dynamic_input)
        return emb_vector
