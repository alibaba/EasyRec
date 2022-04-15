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
from typing import Dict, List, Optional, Callable, Union
import tensorflow as tf
import sparse_operation_kit as sok
from sparse_operation_kit.embeddings.tf_distributed_embedding import TFDistributedEmbedding

Activation = Union[Callable[[tf.Tensor], tf.Tensor], str]

class TFEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 vocab_sizes: Dict[str, int],
                 embedding_vec_size: int,
                 TF_MP: bool = False,
                 comm_options=None,
                 **kwargs):
        super(TFEmbedding, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size
        self._TF_MP = TF_MP
        self._comm_options=comm_options

        self._keras_embedding_layers = {}
        for name, size in self._vocab_sizes.items():
            if self._TF_MP:
                embedding = TFDistributedEmbedding(
                    vocabulary_size=size,
                    embedding_vec_size=self._embedding_vec_size,
                    comm_options=self._comm_options)
            else:
                embedding = tf.keras.layers.Embedding(
                    input_dim=size,
                    output_dim=self._embedding_vec_size)

            self._keras_embedding_layers[name] = embedding
                

    def call(self, inputs: Dict[str, tf.Tensor], training=True) -> Dict[str, tf.Tensor]:
        """
        compute the output of the embedding layer
        """
        output = {}
        for key, val in inputs.items():
            if not isinstance(val, tf.Tensor):
                raise ValueError("Only tf.Tensor is supported for keras Embedding"
                                 f" layers, but got: {type(val)}")

            val = val % self._vocab_sizes[key]
            output[key] = self._keras_embedding_layers[key](val)
        return output


class SOKEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 vocab_sizes: Dict[str, int],
                 embedding_vec_size: int,
                 **kwargs):
        super(SOKEmbedding, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size

        self._sorted_keys = self._vocab_sizes.keys()
        self._vocab_prefix_sum = dict()
        offset = 0
        for key in self._sorted_keys:
            self._vocab_prefix_sum[key] = offset
            offset += self._vocab_sizes[key]
        self._vocab_prefix_sum["total"] = offset
        
        self._sok_embedding = sok.All2AllDenseEmbedding(
                max_vocabulary_size_per_gpu=int(self._vocab_prefix_sum["total"] / 0.75),
                embedding_vec_size=self._embedding_vec_size,
                slot_num=len(self._vocab_sizes),
                nnz_per_slot=1,
                use_hashtable=False)

    def call(self, inputs: Dict[str, tf.Tensor], training=True) -> Dict[str, tf.Tensor]:
        """
        compute the output of the embedding layer
        """
        for key in self._sorted_keys:
            val = inputs[key]
            if not isinstance(val, tf.Tensor):
                raise ValueError("Only tf.Tensor is supported for keras Embedding"
                                f" layers, but got: {type(val)}")

        all_values = tf.stack([tf.math.add(inputs[key], self._vocab_prefix_sum[key]) 
                                for key in self._sorted_keys], axis=1)
        all_emb_vectors = self._sok_embedding(tf.cast(all_values, dtype=tf.int64), 
                                              training=training)
        all_emb_vectors = tf.split(all_emb_vectors, 
                                    num_or_size_splits=len(self._sorted_keys),
                                    axis=1)
        all_emb_vectors = [tf.squeeze(vector) for vector in all_emb_vectors]
        output = dict(zip(self._sorted_keys, all_emb_vectors))
        return output


class MLP(tf.keras.layers.Layer):
    """Sequential multi-layer perceptron (MLP) block."""

    def __init__(self,
                units: List[int],
                use_bias: bool = True,
                activation: Optional[Activation] = "relu",
                final_activation: Optional[Activation] = None,
                **kwargs) -> None:
        """Initializes the MLP layer.
        Args:
        units: Sequential list of layer sizes.
        use_bias: Whether to include a bias term.
        activation: Type of activation to use on all except the last layer.
        final_activation: Type of activation to use on last layer.
        **kwargs: Extra args passed to the Keras Layer base class.
        """

        super().__init__(**kwargs)

        self._sublayers = []

        for num_units in units[:-1]:
            self._sublayers.append(
                tf.keras.layers.Dense(
                    num_units, activation=activation, use_bias=use_bias))
        self._sublayers.append(
            tf.keras.layers.Dense(
                units[-1], activation=final_activation, use_bias=use_bias))

    def call(self, x: tf.Tensor) -> tf.Tensor:
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
                name: Optional[str] = None,
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
        ones = tf.ones_like(xactions)
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
                 vocab_size: List[int],
                 num_dense_features: int,
                 embedding_layer: str,
                 embedding_vec_size: int,
                 bottom_stack_units: List[int],
                 top_stack_units: List[int],
                 TF_MP: bool = False,
                 comm_options=None,
                 **kwargs):
        super(DLRM, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._num_dense_features = num_dense_features
        self._embedding_layer_str = embedding_layer
        self._vocab_size_dict = dict(zip([str(idx) for idx in range(len(self._vocab_size))],
                                         self._vocab_size))
        self._embedding_vec_size = embedding_vec_size
        self._TF_MP = TF_MP
        self._comm_options = comm_options

        if self._embedding_layer_str == "TF":
            self._embedding_layer = TFEmbedding(self._vocab_size_dict, 
                                                self._embedding_vec_size,
                                                TF_MP=self._TF_MP,
                                                comm_options=self._comm_options)
        elif self._embedding_layer_str == "SOK":
            self._embedding_layer = SOKEmbedding(self._vocab_size_dict,
                                                 self._embedding_vec_size)
        else:
            raise ValueError("Not supported embedding_layer. "
                            f"Can only be one of ['TF', 'SOK'], "
                            f"but got {self._embedding_layer_str}")

        self._bottom_stack = MLP(units=bottom_stack_units,
                                 final_activation='relu')

        self._feature_interaction = DotInteraction(skip_gather=True)

        self._top_stack = MLP(units=top_stack_units,
                              final_activation=None)

    @property
    def embedding_layer(self):
        return self._embedding_layer

    def call(self, inputs: Dict[str, tf.Tensor], training=True):
        dense_features = inputs["dense_features"]
        sparse_features = inputs["sparse_features"]

        sparse_embeddings = self._embedding_layer(sparse_features, training=training)
        sparse_embeddings = tf.nest.flatten(sparse_embeddings)
        sparse_embedding_vecs = [
            tf.squeeze(sparse_embedding) for sparse_embedding in sparse_embeddings
        ]

        dense_embedding_vec = self._bottom_stack(dense_features)
        
        interaction_args = sparse_embedding_vecs + [dense_embedding_vec]
        interaction_output = self._feature_interaction(interaction_args)
        feature_interaction_output = tf.concat(
            [dense_embedding_vec, interaction_output], axis=1)

        prediction = self._top_stack(feature_interaction_output)
        return prediction