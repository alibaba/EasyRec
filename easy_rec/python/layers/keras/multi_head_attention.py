# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras import Attention
from easy_rec.python.layers.utils import Parameter


class MultiHeadAttention(Layer):
  """MultiHeadAttention layer.

  This is an implementation of multi-headed attention as described in the
  paper "Attention is all you Need"
  [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
  If `query`, `key,` `value` are the same, then
  this is self-attention. Each time step in `query` attends to the
  corresponding sequence in `key`, and returns a fixed-width vector.

  This layer first projects `query`, `key` and `value`. These are
  (effectively) a list of tensors of length `num_attention_heads`, where the
  corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
  `(batch_size, <key/value dimensions>, key_dim)`,
  `(batch_size, <key/value dimensions>, value_dim)`.

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor.

  Finally, the result tensor with the last dimension as `value_dim` can take
  a linear projection and return.

  Call arguments:
      query: Query tensor of shape `(B, T, dim)`, where `B` is the batch size,
          `T` is the target sequence length, and dim is the feature dimension.
      value: Value tensor of shape `(B, S, dim)`, where `B` is the batch size,
          `S` is the source sequence length, and dim is the feature dimension.
      key: Optional key tensor of shape `(B, S, dim)`. If not given, will
          use `value` for both `key` and `value`, which is the most common
          case.
      training: Python boolean indicating whether the layer should behave in
          training mode (adding dropout) or in inference mode (no dropout).
          Will go with either using the training mode of the parent
          layer/model, or `False` (inference) if there is no parent layer.

  Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
          where `T` is for target sequence shapes and `E` is the query input
          last dimension if `output_shape` is `None`. Otherwise, the
          multi-head outputs are projected to the shape specified by
          `output_shape`.
      attention_scores: (Optional) multi-head attention coefficients over
          attention axes.
  """

  def __init__(self, params, name='attention', reuse=None, **kwargs):
    super(MultiHeadAttention, self).__init__(name=name, **kwargs)
    self.num_heads = params.num_attention_heads
    self.d_model = params.hidden_size
    if self.d_model % self.num_heads != 0:
      raise ValueError(
          'The hidden size (%d) is not a multiple of the number of attention '
          'heads (%d)' % (self.d_model, self.num_heads))
    self.depth = self.d_model // self.num_heads
    self.wq = Dense(self.d_model)
    self.wk = Dense(self.d_model)
    self.wv = Dense(self.d_model)
    self.dense = Dense(self.d_model)
    att_params = Parameter.make_from_pb(params.attention)
    self.attention = Attention(att_params, 'scaled_dot_product_attention')

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs, training=None):
    q, v, k, mask = inputs
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    attn = self.attention([q, v, k], mask=[mask, mask], training=training)
    return_attn_score = self.attention.return_attention_scores
    attention, attention_scores = attn if return_attn_score else attn, None

    attention = tf.transpose(attention, perm=[0, 2, 1, 3])
    attention = tf.reshape(attention, (batch_size, -1, self.d_model))
    output = self.dense(attention)
    if return_attn_score:
      return output, attention_scores
    return output
