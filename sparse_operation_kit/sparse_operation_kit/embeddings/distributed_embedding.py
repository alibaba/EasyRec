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
from sparse_operation_kit.core import SparseEmbeddingLayerHandle
from sparse_operation_kit.embeddings import embedding_ops
import tensorflow as tf

class DistributedEmbedding(tf.keras.layers.Layer):
    """
    Abbreviated as ``sok.DistributedEmbedding(*args, **kwargs)``.

    This is a wrapper class for distributed sparse embedding layer.
    It can be used to create a sparse embedding layer which will distribute
    keys based on `gpu_id = key % gpu_num` to each GPU.

    Parameters
    ----------
    combiner: string
              it is used to specify how to combine embedding vectors intra slots.
              Can be `Mean` or `Sum`.
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
    max_nnz: integer
            the number of maximum valid keys in each slot (feature-filed).
    max_feature_num: integer = slot\_num*max\_nnz
            the maximum valid keys in each sample. It can be used to 
            save GPU memory when this statistic is known. By default, it is equal
            to :math:`max\_feature\_num=slot\_num*max\_nnz`.
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

        emb_layer = sok.DistributedEmbedding(combiner, max_vocabulary_size_per_gpu, 
                                             embedding_vec_size, slot_num, max_nnz,
                                             embedding_initializer=initializer)
        
        @tf.function
        def _train_step(inputs, labels):
            emb_vectors = emb_layer(inputs)
            ...
        
        for i, (inputs, labels) in enumerate(dataset):
            _train_step(inputs)
    """
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size,
                 slot_num,
                 max_nnz,
                 max_feature_num = 1,
                 use_hashtable=True,
                 key_dtype=None,
                 embedding_initializer=None,
                 **kwargs):
        super(DistributedEmbedding, self).__init__(**kwargs)

        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size = embedding_vec_size
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.max_feature_num = max_feature_num

        if self._dtype_policy.variable_dtype is None:
            # in TF1 and policy is not set
            # therefore variable dtype and compute dtype should be fp32
            from tensorflow.python.keras.mixed_precision import experimental as mixed_precision
            self._dtype_policy = mixed_precision.Policy("float32")

        self.var = EmbeddingVariable.CreateInstances(
                                shape=[self.max_vocabulary_size_per_gpu, self.embedding_vec_size],
                                trainable=True,
                                use_hashtable=use_hashtable,
                                dtype=self._dtype_policy.variable_dtype,
                                key_dtype=key_dtype,
                                initializer=embedding_initializer)

        self.emb_layer = SparseEmbeddingLayerHandle(self.var,
                                                    input_dispatcher="all_gather_dispatcher",
                                                    input_dispatcher_subsequent_ops=["csr_conversion_distributed"],
                                                    embedding_executor="distributed",
                                                    output_dispatcher="reduce_scatter_dispatcher",
                                                    slot_num=self.slot_num, 
                                                    max_nnz=self.max_nnz,
                                                    max_feature_num=self.max_feature_num,
                                                    combiner=self.combiner,
                                                    compute_dtype=self._dtype_policy.compute_dtype)

    @property
    def embedding_variable(self):
        return self.var

    def get_config(self):
        config = super(DistributedEmbedding, self).get_config()
        config.update({})
        return config

    def build(self, input_shape):
        pass

    # @tf.function
    def call(self, inputs, training=True):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        inputs: tf.sparse.SparseTensor
                keys are stored in SparseTensor.values. SparseTensor.dense_shape is 
                2-dim and denotes [batchsize * slot_num, max_nnz]. Therefore, the rank
                of SparseTensor.indices must be 2 which denotes [row-indices, column-indices]
                in the corresponding dense tensor.

        training: boolean
                whether training or not.

        Returns
        -------
        emb_vector: tf.float
                the embedding vectors for the input keys. Its shape is
                *[batchsize, slot_num, embedding_vec_size]*
        """
        emb_vector = embedding_ops.embedding_lookup_sparse(
                                        embedding_variable=self.var, 
                                        sp_ids=inputs, 
                                        slot_num=self.slot_num,
                                        training=training)
        return emb_vector
