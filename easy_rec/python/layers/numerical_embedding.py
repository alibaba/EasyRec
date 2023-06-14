# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import tensorflow as tf

from easy_rec.python.utils.activation import get_activation

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class NLinear(object):
  """N linear layers for N token (feature) embeddings.

  To understand this module, let's revise `tf.layers.dense`. When `tf.layers.dense` is
  applied to three-dimensional inputs of the shape
  ``(batch_size, n_tokens, d_embedding)``, then the same linear transformation is
  applied to each of ``n_tokens`` token (feature) embeddings.

  By contrast, `NLinear` allocates one linear layer per token (``n_tokens`` layers in total).
  One such layer can be represented as ``tf.layers.dense(d_in, d_out)``.
  So, the i-th linear transformation is applied to the i-th token embedding, as
  illustrated in the following pseudocode::

      layers = [tf.layers.dense(d_in, d_out) for _ in range(n_tokens)]
      x = tf.random.normal(batch_size, n_tokens, d_in)
      result = tf.stack([layers[i](x[:, i]) for i in range(n_tokens)], 1)

  Examples:
      .. testcode::

          batch_size = 2
          n_features = 3
          d_embedding_in = 4
          d_embedding_out = 5
          x = tf.random.normal(batch_size, n_features, d_embedding_in)
          m = NLinear(n_features, d_embedding_in, d_embedding_out)
          assert m(x).shape == (batch_size, n_features, d_embedding_out)
  """

  def __init__(self, n_tokens, d_in, d_out, bias=True, scope='nd_linear'):
    """Init with input shapes.

    Args:
        n_tokens: the number of tokens (features)
        d_in: the input dimension
        d_out: the output dimension
        bias: indicates if the underlying linear layers have biases
    """
    with tf.variable_scope(scope):
      self.weight = tf.get_variable(
          'weights', [1, n_tokens, d_in, d_out], dtype=tf.float32)
      if bias:
        initializer = tf.constant_initializer(0.0)
        self.bias = tf.get_variable(
            'bias', [1, n_tokens, d_out],
            dtype=tf.float32,
            initializer=initializer)
      else:
        self.bias = None

  def __call__(self, x, *args, **kwargs):
    if x.shape.ndims != 3:
      raise ValueError(
          'The input must have three dimensions (batch_size, n_tokens, d_embedding)'
      )
    if x.shape[2] != self.weight.shape[2]:
      raise ValueError('invalid input embedding dimension %d, expect %d' %
                       (int(x.shape[2]), int(self.weight.shape[2])))

    x = x[..., None] * self.weight  # [B, N, D, D_out]
    x = tf.reduce_sum(x, axis=-2)  # [B, N, D_out]
    if self.bias is not None:
      x = x + self.bias
    return x


class PeriodicEmbedding(object):
  """Periodic embeddings for numerical features described in [1].

  References:
    * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko,
    "On Embeddings for Numerical Features in Tabular Deep Learning", 2022
    https://arxiv.org/pdf/2203.05556.pdf
  """

  def __init__(self, config, scope='periodic_embedding'):
    """Init with a pb config.

    Args:
      config: pb config
      config.embedding_dim: the embedding size, must be an even positive integer.
      config.sigma: the scale of the weight initialization.
        **This is a super important parameter which significantly affects performance**.
        Its optimal value can be dramatically different for different datasets, so
        no "default value" can exist for this parameter, and it must be tuned for
        each dataset. In the original paper, during hyperparameter tuning, this
        parameter was sampled from the distribution ``LogUniform[1e-2, 1e2]``.
        A similar grid would be ``[1e-2, 1e-1, 1e0, 1e1, 1e2]``.
        If possible, add more intermidiate values to this grid.
      config.output_3d_tensor: whether to output a 3d tensor
    """
    self.config = config
    if config.embedding_dim % 2:
      raise ValueError('embedding_dim must be even')
    self.emb_dim = config.embedding_dim // 2
    self.scope = scope
    self.initializer = tf.random_normal_initializer(stddev=config.sigma)

  def __call__(self, inputs, *args, **kwargs):
    if inputs.shape.ndims != 2:
      raise ValueError('inputs of PeriodicEmbedding must have 2 dimensions.')

    num_features = int(inputs.shape[-1])
    with tf.variable_scope(self.scope):
      c = tf.get_variable(
          'coefficients',
          shape=[1, num_features, self.emb_dim],
          initializer=self.initializer)

      features = inputs[..., None]  # [B, N, 1]
      v = 2 * math.pi * c * features  # [B, N, E]
      emb = tf.concat([tf.sin(v), tf.cos(v)], axis=-1)  # [B, N, 2E]

      dim = self.config.embedding_dim
      if self.config.add_linear_layer:
        linear = NLinear(num_features, dim, dim)
        emb = linear(emb)
        act = get_activation(self.config.linear_activation)
        if callable(act):
          emb = act(emb)

      if self.config.output_3d_tensor:
        return emb
      return tf.reshape(emb, [-1, num_features * dim])


class AutoDisEmbedding(object):

  def __init__(self, config, scope='auto_dis'):
    """An Embedding Learning Framework for Numerical Features in CTR Prediction.

    Refer: https://arxiv.org/pdf/2012.08986v2.pdf
    """
    self.config = config
    self.emb_dim = config.embedding_dim
    self.num_bins = config.num_bins
    self.scope = scope

  def __call__(self, inputs, *args, **kwargs):
    if inputs.shape.ndims != 2:
      raise ValueError('inputs of PeriodicEmbedding must have 2 dimensions.')

    num_features = int(inputs.shape[-1])
    with tf.variable_scope(self.scope):
      meta_emb = tf.get_variable(
          'meta_embedding',
          shape=[1, num_features, self.num_bins, self.emb_dim])
      w = tf.get_variable('project_w', shape=[1, num_features, self.num_bins])
      mat = tf.get_variable(
          'project_mat', shape=[1, num_features, self.num_bins, self.num_bins])

      x = tf.expand_dims(inputs, axis=-1)  # [B, num_fea, 1]
      hidden = tf.nn.leaky_relu(w * x)  # [B, num_fea, num_bin]

      y = tf.matmul(mat, hidden[..., None])  # [B, num_fea, num_bin, 1]
      y = tf.squeeze(y, axis=3)  # [B, num_fea, num_bin]

      # keep_prob(float): if dropout_flag is True, keep_prob rate to keep connect; (float, keep_prob=0.8)
      alpha = self.config.keep_prob
      x_bar = y + alpha * hidden  # [B, num_fea, num_bin]
      t = self.config.temperature
      x_hat = tf.nn.softmax(x_bar / t)  # [B, num_fea, num_bin]

      emb = tf.matmul(x_hat[:, :, None, :], meta_emb)  # [B, num_fea, 1, emb_dim]
      # emb = tf.squeeze(emb, axis=2)  # [B, num_fea, emb_dim]
      if self.config.output_3d_tensor:
        return tf.reshape(
            emb, [-1, num_features, self.emb_dim])  # [B, num_fea, emb_dim]
      return tf.reshape(
          emb, [-1, self.emb_dim * num_features])  # [B, num_fea*emb_dim]
