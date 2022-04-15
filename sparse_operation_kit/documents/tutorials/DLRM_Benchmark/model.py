import os
import math
from typing import Dict, List, Optional

import sparse_operation_kit as sok
import tensorflow as tf
import numpy as np

try:
    from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops
except:
    pass


class SOKEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 vocab_sizes: List[int],
                 embedding_vec_size: int,
                 num_gpus=1,
                 **kwargs):
        super(SOKEmbedding, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size

        prefix_sum = []
        offset = 0
        for i in range(len(vocab_sizes)):
            prefix_sum.append(offset)
            offset += self._vocab_sizes[i]
        prefix_sum = np.array(prefix_sum, dtype=np.int64).reshape(1, -1)
        self._vocab_prefix_sum = tf.constant(prefix_sum)
        print('[Info] Total vocabulary size:', offset)

        self._sok_embedding = sok.All2AllDenseEmbedding(
            max_vocabulary_size_per_gpu=int(offset / num_gpus + 1),
            embedding_vec_size=self._embedding_vec_size,
            slot_num=len(self._vocab_sizes),
            nnz_per_slot=1,
            dynamic_input=True,
            use_hashtable=False,
        )

    def call(self, inputs, training=True, compress=False):
        """
        Compute the output of embedding layer.
        Args:
            inputs: [batch_size, 26] int64 tensor
        Returns:
            emb_vectors: [batch_size, 26, embedding_vec_size] float32 tensor
        """
        # inputs                : [batch_size, 26] int64 tensor
        # self._vocab_prefix_sum: [1, 26] int64 tensor
        # fused_inputs          : [batch_size, 26] int64 tensor
        fused_inputs = tf.add(inputs, self._vocab_prefix_sum)

        # fused_inputs: [batch_size*26] int64 tensor
        fused_inputs = tf.reshape(fused_inputs, [-1])

        if compress:
            # Make sure there are no duplicate items in fused_inputs, to reduce communication overhead
            fused_inputs, idx = tf.unique(fused_inputs)

        # emb_vectors: [batch_size*26, embedding_vec_size]
        emb_vectors = self._sok_embedding(fused_inputs, training=training)

        if compress:
            # Restore the first dimension of emb_vectors to batch_size*26
            # Used in pairs with tf.unique()
            emb_vectors = tf.gather(emb_vectors, idx)

        # emb_vectors: [batch_size, 26, embedding_vec_size]
        emb_vectors = tf.reshape(emb_vectors, [-1, len(self._vocab_sizes), self._embedding_vec_size])

        return emb_vectors


class TFEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_sizes,
                 embedding_vec_size,
                 num_gpus=1,
                 **kwargs):
        super(TFEmbedding, self).__init__(**kwargs)
        assert(num_gpus == 1)
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size

        self._tf_embeddings = []
        for size in vocab_sizes:
            emb = tf.keras.layers.Embedding(
                input_dim=size,
                output_dim=self._embedding_vec_size,
            )
            self._tf_embeddings.append(emb)

    def call(self, inputs, training=True):
        fused_inputs = tf.split(inputs, num_or_size_splits=len(self._vocab_sizes), axis=1)
        emb_vectors = []
        for i in range(len(self._tf_embeddings)):
            out = self._tf_embeddings[i](fused_inputs[i])
            emb_vectors.append(out)
        emb_vectors = tf.concat(emb_vectors, axis=1)
        return emb_vectors


class MLP(tf.keras.layers.Layer):
    """Sequential multi-layer perceptron (MLP) block."""

    def __init__(self,
                units: List[int],
                use_bias: bool = True,
                activation="relu",
                final_activation=None,
                **kwargs):
        """Initializes the MLP layer.
        Args:
        units: Sequential list of layer sizes. List[int].
        use_bias: Whether to include a bias term.
        activation: Type of activation to use on all except the last layer.
        final_activation: Type of activation to use on last layer.
        **kwargs: Extra args passed to the Keras Layer base class.
        """
        super().__init__(**kwargs)

        self._sublayers = []
        for i, num_units in enumerate(units):
            kernel_init = tf.keras.initializers.GlorotNormal()
            bias_init = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1. / num_units))

            self._sublayers.append(
                tf.keras.layers.Dense(
                    num_units, use_bias=use_bias,
                    kernel_initializer=kernel_init, bias_initializer=bias_init,
                    activation=activation if i < (len(units) - 1) else final_activation))

    def call(self, x: tf.Tensor):
        """Performs the forward computation of the block."""
        for layer in self._sublayers:
            x = layer(x)
        return x


class DotInteraction(tf.keras.layers.Layer):
    """Dot interaction layer.
    See theory in the DLRM paper: https://arxiv.org/pdf/1906.00091.pdf,
    section 2.1.3. Sparse activations and dense activations are combined.
    Dot interaction is applied to a batch of input Tensors [e1,...,e_k] of the
    same dimension and the output is a batch of Tensors with all distinct pairwise
    dot products of the form dot(e_i, e_j) for i <= j if self self_interaction is
    True, otherwise dot(e_i, e_j) i < j.
    Attributes:
        self_interaction: Boolean indicating if features should self-interact.
        If it is True, then the diagonal enteries of the interaction matric are
        also taken.
        skip_gather: An optimization flag. If it's set then the upper triangle part
        of the dot interaction matrix dot(e_i, e_j) is set to 0. The resulting
        activations will be of dimension [num_features * num_features] from which
        half will be zeros. Otherwise activations will be only lower triangle part
        of the interaction matrix. The later saves space but is much slower.
        name: String name of the layer.
    """

    def __init__(self,
                self_interaction: bool = False,
                skip_gather: bool = False,
                name=None,
                **kwargs) -> None:
        self._self_interaction = self_interaction
        self._skip_gather = skip_gather
        super().__init__(name=name, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """Performs the interaction operation on the tensors in the list.
        The tensors represent as transformed dense features and embedded categorical
        features.
        Pre-condition: The tensors should all have the same shape.
        Args:
        inputs: List of features with shapes [batch_size, feature_dim].
        Returns:
        activations: Tensor representing interacted features. It has a dimension
        `num_features * num_features` if skip_gather is True, otherside
        `num_features * (num_features + 1) / 2` if self_interaction is True and
        `num_features * (num_features - 1) / 2` if self_interaction is False.
        """
        num_features = len(inputs)
        batch_size = tf.shape(inputs[0])[0]
        feature_dim = tf.shape(inputs[0])[1]
        # concat_features shape: batch_size, num_features, feature_dim
        try:
            concat_features = tf.concat(inputs, axis=-1)
            concat_features = tf.reshape(concat_features,
                                        [batch_size, -1, feature_dim])
        except (ValueError, tf.errors.InvalidArgumentError) as e:
            raise ValueError(f"Input tensors` dimensions must be equal, original"
                        f"error message: {e}")

        # Interact features, select lower-triangular portion, and re-shape.
        xactions = tf.matmul(concat_features, concat_features, transpose_b=True)
        ones = tf.ones_like(xactions, dtype=tf.float32)
        if self._self_interaction:
            # Selecting lower-triangular portion including the diagonal.
            lower_tri_mask = tf.linalg.band_part(ones, -1, 0)
            upper_tri_mask = ones - lower_tri_mask
            out_dim = num_features * (num_features + 1) // 2
        else:
            # Selecting lower-triangular portion not included the diagonal.
            upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
            lower_tri_mask = ones - upper_tri_mask
            out_dim = num_features * (num_features - 1) // 2

        if self._skip_gather:
            # Setting upper tiangle part of the interaction matrix to zeros.
            activations = tf.where(condition=tf.cast(upper_tri_mask, tf.bool),
                                    x=tf.zeros_like(xactions),
                                    y=xactions)
            out_dim = num_features * num_features
        else:
            activations = tf.boolean_mask(xactions, lower_tri_mask)

        activations = tf.reshape(activations, (batch_size, out_dim))
        return activations


class DLRM(tf.keras.models.Model):
    def __init__(self,
                 vocab_sizes: List[int],
                 num_dense_features: int,
                 embedding_vec_size: int,
                 bottom_stack_units: List[int],
                 top_stack_units: List[int],
                 num_gpus=1,
                 use_cuda_interact=False,
                 compress=False,
                 **kwargs):
        super(DLRM, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._num_dense_features = num_dense_features
        self._embedding_vec_size = embedding_vec_size
        self._use_cuda_interact = use_cuda_interact
        self._compress = compress

        self._embedding_layer = SOKEmbedding(self._vocab_sizes,
                                             self._embedding_vec_size,
                                             num_gpus)
        self._bottom_stack = MLP(units=bottom_stack_units,
                                 final_activation='relu')
        self._feature_interaction = DotInteraction(self_interaction=False, skip_gather=False)
        self._top_stack = MLP(units=top_stack_units,
                              final_activation=None)

    def call(self, inputs: List[tf.Tensor], training=True):
        dense_input = inputs[0]
        # if amp:
        #     dense_input = tf.cast(dense_input, tf.float16)
        dense_embedding_vec = self._bottom_stack(dense_input)
        sparse_embeddings = self._embedding_layer(inputs[1], training=training, compress=self._compress)

        if not self._use_cuda_interact:
            sparse_embeddings = tf.split(sparse_embeddings,
                                        num_or_size_splits=len(self._vocab_sizes),
                                        axis=1)
            sparse_embedding_vecs = [tf.squeeze(vec) for vec in sparse_embeddings]
            interaction_args = sparse_embedding_vecs + [dense_embedding_vec]
            interaction_output = self._feature_interaction(interaction_args)
            feature_interaction_output = tf.concat(
                [dense_embedding_vec, interaction_output], axis=1)
        else:
            try:
                dense_embedding = tf.expand_dims(dense_embedding_vec, 1)
                interact_args = tf.concat([dense_embedding, sparse_embeddings], axis=1)
                feature_interaction_output = dot_based_interact_ops.dot_based_interact(interact_args, dense_embedding_vec)
            except:
                raise RuntimeError('tensorflow_dot_based_interact is not installed.')

        prediction = self._top_stack(feature_interaction_output)
        return prediction
