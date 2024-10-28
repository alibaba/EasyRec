# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import math
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers import Layer

from easy_rec.python.compat.array_ops import repeat
from easy_rec.python.utils.activation import get_activation
from easy_rec.python.utils.tf_utils import get_ps_num_from_tf_config

curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
ops_idr = os.path.dirname(parent_dir)
ops_dir = os.path.join(ops_idr, 'ops')
if 'PAI' in tf.__version__:
  ops_dir = os.path.join(ops_dir, '1.12_pai')
elif tf.__version__.startswith('1.12'):
  ops_dir = os.path.join(ops_dir, '1.12')
elif tf.__version__.startswith('1.15'):
  if 'IS_ON_PAI' in os.environ:
    ops_dir = os.path.join(ops_dir, 'DeepRec')
  else:
    ops_dir = os.path.join(ops_dir, '1.15')
elif tf.__version__.startswith('2.12'):
  ops_dir = os.path.join(ops_dir, '2.12')

logging.info('ops_dir is %s' % ops_dir)
custom_op_path = os.path.join(ops_dir, 'libcustom_ops.so')
try:
  custom_ops = tf.load_op_library(custom_op_path)
  logging.info('load custom op from %s succeed' % custom_op_path)
except Exception as ex:
  logging.warning('load custom op from %s failed: %s' %
                  (custom_op_path, str(ex)))
  custom_ops = None


class NLinear(Layer):
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

  def __init__(self,
               n_tokens,
               d_in,
               d_out,
               bias=True,
               name='nd_linear',
               **kwargs):
    """Init with input shapes.

    Args:
        n_tokens: the number of tokens (features)
        d_in: the input dimension
        d_out: the output dimension
        bias: indicates if the underlying linear layers have biases
        name: layer name
    """
    super(NLinear, self).__init__(name=name, **kwargs)
    self.weight = self.add_weight(
        'weights', [1, n_tokens, d_in, d_out], dtype=tf.float32)
    if bias:
      initializer = tf.constant_initializer(0.0)
      self.bias = self.add_weight(
          'bias', [1, n_tokens, d_out],
          dtype=tf.float32,
          initializer=initializer)
    else:
      self.bias = None

  def call(self, x, **kwargs):
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


class PeriodicEmbedding(Layer):
  """Periodic embeddings for numerical features described in [1].

  References:
    * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko,
    "On Embeddings for Numerical Features in Tabular Deep Learning", 2022
    https://arxiv.org/pdf/2203.05556.pdf

  Attributes:
    embedding_dim: the embedding size, must be an even positive integer.
    sigma: the scale of the weight initialization.
      **This is a super important parameter which significantly affects performance**.
      Its optimal value can be dramatically different for different datasets, so
      no "default value" can exist for this parameter, and it must be tuned for
      each dataset. In the original paper, during hyperparameter tuning, this
      parameter was sampled from the distribution ``LogUniform[1e-2, 1e2]``.
      A similar grid would be ``[1e-2, 1e-1, 1e0, 1e1, 1e2]``.
      If possible, add more intermediate values to this grid.
    output_3d_tensor: whether to output a 3d tensor
    output_tensor_list: whether to output the list of embedding
  """

  def __init__(self, params, name='periodic_embedding', reuse=None, **kwargs):
    super(PeriodicEmbedding, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    params.check_required(['embedding_dim', 'sigma'])
    self.embedding_dim = int(params.embedding_dim)
    if self.embedding_dim % 2:
      raise ValueError('embedding_dim must be even')
    sigma = params.sigma
    self.initializer = tf.random_normal_initializer(stddev=sigma)
    self.add_linear_layer = params.get_or_default('add_linear_layer', True)
    self.linear_activation = params.get_or_default('linear_activation', 'relu')
    self.output_tensor_list = params.get_or_default('output_tensor_list', False)
    self.output_3d_tensor = params.get_or_default('output_3d_tensor', False)

  def build(self, input_shape):
    if input_shape.ndims != 2:
      raise ValueError('inputs of AutoDisEmbedding must have 2 dimensions.')
    self.num_features = int(input_shape[-1])
    num_ps = get_ps_num_from_tf_config()
    partitioner = None
    if num_ps > 0:
      partitioner = tf.fixed_size_partitioner(num_shards=num_ps)
    emb_dim = self.embedding_dim // 2
    self.coef = self.add_weight(
        'coefficients',
        shape=[1, self.num_features, emb_dim],
        partitioner=partitioner,
        initializer=self.initializer)
    if self.add_linear_layer:
      self.linear = NLinear(
          self.num_features,
          self.embedding_dim,
          self.embedding_dim,
          name='nd_linear')
    super(PeriodicEmbedding, self).build(input_shape)

  def call(self, inputs, **kwargs):
    features = inputs[..., None]  # [B, N, 1]
    v = 2 * math.pi * self.coef * features  # [B, N, E]
    emb = tf.concat([tf.sin(v), tf.cos(v)], axis=-1)  # [B, N, 2E]

    dim = self.embedding_dim
    if self.add_linear_layer:
      emb = self.linear(emb)
      act = get_activation(self.linear_activation)
      if callable(act):
        emb = act(emb)
    output = tf.reshape(emb, [-1, self.num_features * dim])

    if self.output_tensor_list:
      return output, tf.unstack(emb, axis=1)
    if self.output_3d_tensor:
      return output, emb
    return output


class AutoDisEmbedding(Layer):
  """An Embedding Learning Framework for Numerical Features in CTR Prediction.

  Refer: https://arxiv.org/pdf/2012.08986v2.pdf
  """

  def __init__(self, params, name='auto_dis_embedding', reuse=None, **kwargs):
    super(AutoDisEmbedding, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    params.check_required(['embedding_dim', 'num_bins', 'temperature'])
    self.emb_dim = int(params.embedding_dim)
    self.num_bins = int(params.num_bins)
    self.temperature = params.temperature
    self.keep_prob = params.get_or_default('keep_prob', 0.8)
    self.output_tensor_list = params.get_or_default('output_tensor_list', False)
    self.output_3d_tensor = params.get_or_default('output_3d_tensor', False)

  def build(self, input_shape):
    if input_shape.ndims != 2:
      raise ValueError('inputs of AutoDisEmbedding must have 2 dimensions.')
    self.num_features = int(input_shape[-1])
    num_ps = get_ps_num_from_tf_config()
    partitioner = None
    if num_ps > 0:
      partitioner = tf.fixed_size_partitioner(num_shards=num_ps)
    self.meta_emb = self.add_weight(
        'meta_embedding',
        shape=[self.num_features, self.num_bins, self.emb_dim],
        partitioner=partitioner)
    self.proj_w = self.add_weight(
        'project_w',
        shape=[1, self.num_features, self.num_bins],
        partitioner=partitioner)
    self.proj_mat = self.add_weight(
        'project_mat',
        shape=[self.num_features, self.num_bins, self.num_bins],
        partitioner=partitioner)
    super(AutoDisEmbedding, self).build(input_shape)

  def call(self, inputs, **kwargs):
    x = tf.expand_dims(inputs, axis=-1)  # [B, N, 1]
    hidden = tf.nn.leaky_relu(self.proj_w * x)  # [B, N, num_bin]
    # 低版本的tf(1.12) matmul 不支持广播，所以改成 einsum
    # y = tf.matmul(mat, hidden[..., None])  # [B, N, num_bin, 1]
    # y = tf.squeeze(y, axis=3)  # [B, N, num_bin]
    y = tf.einsum('nik,bnk->bni', self.proj_mat, hidden)  # [B, N, num_bin]

    # keep_prob(float): if dropout_flag is True, keep_prob rate to keep connect
    alpha = self.keep_prob
    x_bar = y + alpha * hidden  # [B, N, num_bin]
    x_hat = tf.nn.softmax(x_bar / self.temperature)  # [B, N, num_bin]

    # emb = tf.matmul(x_hat[:, :, None, :], meta_emb)  # [B, N, 1, D]
    # emb = tf.squeeze(emb, axis=2)  # [B, N, D]
    emb = tf.einsum('bnk,nkd->bnd', x_hat, self.meta_emb)
    output = tf.reshape(emb, [-1, self.emb_dim * self.num_features])  # [B, N*D]

    if self.output_tensor_list:
      return output, tf.unstack(emb, axis=1)
    if self.output_3d_tensor:
      return output, emb
    return output


class NaryDisEmbedding(Layer):
  """Numerical Feature Representation with Hybrid 𝑁 -ary Encoding, CIKM 2022..

  Refer: https://dl.acm.org/doi/pdf/10.1145/3511808.3557090
  """

  def __init__(self, params, name='nary_dis_embedding', reuse=None, **kwargs):
    super(NaryDisEmbedding, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self.nary_carry = custom_ops.nary_carry
    params.check_required(['embedding_dim', 'carries'])
    self.emb_dim = int(params.embedding_dim)
    self.carries = params.get_or_default('carries', [2, 9])
    self.num_replicas = params.get_or_default('num_replicas', 1)
    assert self.num_replicas >= 1, 'num replicas must be >= 1'
    self.lengths = list(map(self.max_length, self.carries))
    self.vocab_size = int(sum(self.lengths))
    self.multiplier = params.get_or_default('multiplier', 1.0)
    self.intra_ary_pooling = params.get_or_default('intra_ary_pooling', 'sum')
    self.output_3d_tensor = params.get_or_default('output_3d_tensor', False)
    self.output_tensor_list = params.get_or_default('output_tensor_list', False)
    logging.info(
        '{} carries: {}, lengths: {}, vocab_size: {}, intra_ary: {}, replicas: {}, multiplier: {}'
        .format(self.name, ','.join(map(str, self.carries)),
                ','.join(map(str, self.lengths)), self.vocab_size,
                self.intra_ary_pooling, self.num_replicas, self.multiplier))

  @staticmethod
  def max_length(carry):
    bits = math.log(4294967295, carry)
    return (math.floor(bits) + 1) * carry

  def build(self, input_shape):
    assert isinstance(input_shape,
                      tf.TensorShape), 'NaryDisEmbedding only takes 1 input'
    self.num_features = int(input_shape[-1])
    logging.info('%s has %d input features', self.name, self.num_features)
    vocab_size = self.num_features * self.vocab_size
    emb_dim = self.emb_dim * self.num_replicas
    num_ps = get_ps_num_from_tf_config()
    partitioner = None
    if num_ps > 0:
      partitioner = tf.fixed_size_partitioner(num_shards=num_ps)
    self.embedding_table = self.add_weight(
        'embed_table', shape=[vocab_size, emb_dim], partitioner=partitioner)
    super(NaryDisEmbedding, self).build(input_shape)

  def call(self, inputs, **kwargs):
    if inputs.shape.ndims != 2:
      raise ValueError('inputs of NaryDisEmbedding must have 2 dimensions.')
    if self.multiplier != 1.0:
      inputs *= self.multiplier
    inputs = tf.to_int32(inputs)
    offset, emb_indices, emb_splits = 0, [], []
    with ops.device('/CPU:0'):
      for carry, length in zip(self.carries, self.lengths):
        values, splits = self.nary_carry(inputs, carry=carry, offset=offset)
        offset += length
        emb_indices.append(values)
        emb_splits.append(splits)
    indices = tf.concat(emb_indices, axis=0)
    splits = tf.concat(emb_splits, axis=0)
    # embedding shape: [B*N*C, D]
    embedding = tf.nn.embedding_lookup(self.embedding_table, indices)

    total_length = tf.size(splits)
    if self.intra_ary_pooling == 'sum':
      if tf.__version__ >= '2.0':
        segment_ids = tf.repeat(tf.range(total_length), repeats=splits)
      else:
        segment_ids = repeat(tf.range(total_length), repeats=splits)
      embedding = tf.math.segment_sum(embedding, segment_ids)
    elif self.intra_ary_pooling == 'mean':
      if tf.__version__ >= '2.0':
        segment_ids = tf.repeat(tf.range(total_length), repeats=splits)
      else:
        segment_ids = repeat(tf.range(total_length), repeats=splits)
      embedding = tf.math.segment_mean(embedding, segment_ids)
    else:
      raise ValueError('Unsupported intra ary pooling method %s' %
                       self.intra_ary_pooling)
    # B: batch size
    # N: num features
    # C: num carries
    # D: embedding dimension
    # R: num replicas
    # shape of embedding: [B*N*C, R*D]
    N = self.num_features
    C = len(self.carries)
    D = self.emb_dim
    if self.num_replicas == 1:
      embedding = tf.reshape(embedding, [C, -1, D])  # [C, B*N, D]
      embedding = tf.transpose(embedding, perm=[1, 0, 2])  # [B*N, C, D]
      embedding = tf.reshape(embedding, [-1, C * D])  # [B*N, C*D]
      output = tf.reshape(embedding, [-1, N * C * D])  # [B, N*C*D]
      if self.output_tensor_list:
        return output, tf.split(embedding, N)  # [B, C*D] * N
      if self.output_3d_tensor:
        embedding = tf.reshape(embedding, [-1, N, C * D])  # [B, N, C*D]
        return output, embedding
      return output

    # self.num_replicas > 1:
    replicas = tf.split(embedding, self.num_replicas, axis=1)
    outputs = []
    outputs2 = []
    for replica in replicas:
      # shape of replica: [B*N*C, D]
      embedding = tf.reshape(replica, [C, -1, D])  # [C, B*N, D]
      embedding = tf.transpose(embedding, perm=[1, 0, 2])  # [B*N, C, D]
      embedding = tf.reshape(embedding, [-1, C * D])  # [B*N, C*D]
      output = tf.reshape(embedding, [-1, N * C * D])  # [B, N*C*D]
      outputs.append(output)
      if self.output_tensor_list:
        embedding = tf.split(embedding, N)  # [B, C*D] * N
        outputs2.append(embedding)
      elif self.output_3d_tensor:
        embedding = tf.reshape(embedding, [-1, N, C * D])  # [B, N, C*D]
        outputs2.append(embedding)
    return outputs + outputs2
