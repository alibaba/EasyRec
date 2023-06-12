# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import logging
import six
import tensorflow as tf

from easy_rec.python.compat.layers import layer_norm as tf_layer_norm
from easy_rec.python.utils.activation import get_activation

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def highway(x,
            size=None,
            activation=None,
            num_layers=1,
            scope='highway',
            dropout=0.0,
            reuse=None):
  if isinstance(activation, six.string_types):
    activation = get_activation(activation)
  with tf.variable_scope(scope, reuse):
    if size is None:
      size = x.shape.as_list()[-1]
    else:
      x = tf.layers.dense(x, size, name='input_projection', reuse=reuse)

    for i in range(num_layers):
      T = tf.layers.dense(
          x, size, activation=tf.sigmoid, name='gate_%d' % i, reuse=reuse)
      H = tf.layers.dense(
          x, size, activation=activation, name='activation_%d' % i, reuse=reuse)
      if dropout > 0.0:
        H = tf.nn.dropout(H, 1.0 - dropout)
      x = H * T + x * (1.0 - T)
    return x


def text_cnn(x,
             filter_sizes=(3, 4, 5),
             num_filters=(128, 64, 64),
             scope_name='textcnn',
             reuse=False):
  # x: None * step_dim * embed_dim
  assert len(filter_sizes) == len(num_filters)
  initializer = tf.variance_scaling_initializer()
  pooled_outputs = []
  for i in range(len(filter_sizes)):
    filter_size = filter_sizes[i]
    num_filter = num_filters[i]
    scope_name_i = scope_name + '_' + str(filter_size)
    with tf.variable_scope(scope_name_i, reuse=reuse):
      # conv shape: (batch_size, seq_len - filter_size + 1, num_filters)
      conv = tf.layers.conv1d(
          x,
          filters=int(num_filter),
          kernel_size=int(filter_size),
          activation=tf.nn.relu,
          name='conv_layer',
          reuse=reuse,
          kernel_initializer=initializer,
          padding='same')
      pool = tf.reduce_max(
          conv, axis=1)  # max pooling, shape: (batch_size, num_filters)
    pooled_outputs.append(pool)
  pool_flat = tf.concat(
      pooled_outputs, 1)  # shape: (batch_size, num_filters * len(filter_sizes))
  return pool_flat


def layer_norm(input_tensor, name=None, reuse=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf_layer_norm(
      inputs=input_tensor,
      begin_norm_axis=-1,
      begin_params_axis=-1,
      reuse=reuse,
      scope=name)


class SENet(object):
  """SENet+ Layer used in FiBiNET，支持不同field的embedding dimension不等.

  arxiv: 2209.05016
  """

  def __init__(self, config, name='SENet'):
    self.config = config
    self.name = name

  def __call__(self, embedding_list):
    """embedding_list:  - A list of 2D tensor with shape: ``(batch_size,embedding_size)``."""
    print('SENET layer with %d inputs' % len(embedding_list))
    g = self.config.num_squeeze_group
    for emb in embedding_list:
      assert emb.shape.ndims == 2, 'field embeddings must be rank 2 tensors'
      dim = int(emb.shape[-1])
      assert dim >= g and dim % g == 0, 'field embedding dimension %d must be divisible by %d' % (
          dim, g)

    field_size = len(embedding_list)
    feature_size_list = [emb.shape.as_list()[-1] for emb in embedding_list]

    # Squeeze
    # embedding dimension 必须能被 g 整除
    group_embs = [
        tf.reshape(emb, [-1, g, int(emb.shape[-1]) // g])
        for emb in embedding_list
    ]

    squeezed = []
    for emb in group_embs:
      squeezed.append(tf.reduce_max(emb, axis=-1))  # [B, g]
      squeezed.append(tf.reduce_mean(emb, axis=-1))  # [B, g]
    z = tf.concat(squeezed, axis=1)  # [bs, field_size * num_groups * 2]

    # Excitation
    r = self.config.reduction_ratio
    reduction_size = max(1, field_size * g * 2 // r)

    initializer = tf.glorot_normal_initializer()
    a1 = tf.layers.dense(
        z,
        reduction_size,
        kernel_initializer=initializer,
        activation=tf.nn.relu,
        name='%s/W1' % self.name)
    weights = tf.layers.dense(
        a1,
        sum(feature_size_list),
        kernel_initializer=initializer,
        name='%s/W2' % self.name)

    # Re-weight
    inputs = tf.concat(embedding_list, axis=-1)
    output = inputs * weights

    # Fuse, add skip-connection
    if self.config.use_skip_connection:
      output += inputs

    # Layer Normalization
    if self.config.use_output_layer_norm:
      output = layer_norm(output)
    return output


def _full_interaction(v_i, v_j):
  # [bs, 1, dim] x [bs, dim, 1] = [bs, 1]
  interaction = tf.matmul(
      tf.expand_dims(v_i, axis=1), tf.expand_dims(v_j, axis=-1))
  return tf.squeeze(interaction, axis=1)


class BiLinear(object):

  def __init__(self,
               output_size,
               bilinear_type,
               bilinear_plus=True,
               name='bilinear'):
    """双线性特征交互层，支持不同field embeddings的size不等.

    arxiv: 2209.05016
    :param output_size: 输出的size
    :param bilinear_type: ['all', 'each', 'interaction']，支持其中一种
    :param bilinear_plus: 是否使用bi-linear+
    """
    self.name = name
    self.bilinear_type = bilinear_type.lower()
    self.output_size = output_size

    if bilinear_type not in ['all', 'each', 'interaction']:
      raise NotImplementedError(
          "bilinear_type only support: ['all', 'each', 'interaction']")

    if bilinear_plus:
      self.func = _full_interaction
    else:
      self.func = tf.multiply

  def __call__(self, embeddings):
    print('Bilinear Layer with %d inputs' % len(embeddings))
    if len(embeddings) > 200:
      logging.warn('There are too many inputs for bilinear layer: %d' %
                   len(embeddings))
    equal_dim = True
    _dim = embeddings[0].shape[-1]
    for emb in embeddings:
      assert emb.shape.ndims == 2, 'field embeddings must be rank 2 tensors'
      if emb.shape[-1] != _dim:
        equal_dim = False
    if not equal_dim and self.bilinear_type != 'interaction':
      raise ValueError(
          'all embedding dimensions must be same when not use bilinear type: interaction'
      )
    dim = int(_dim)

    field_size = len(embeddings)
    initializer = tf.glorot_normal_initializer()

    # bi-linear+: p的维度为[bs, f*(f-1)/2]
    # bi-linear:
    # 当equal_dim=True时，p的维度为[bs, f*(f-1)/2*k]，k为embeddings的size
    # 当equal_dim=False时，p的维度为[bs, (k_2+k_3+...+k_f)+...+(k_i+k_{i+1}+...+k_f)+...+k_f]，
    # 其中 k_i为第i个field的embedding的size
    if self.bilinear_type == 'all':
      v_dot = [
          tf.layers.dense(
              v_i,
              dim,
              kernel_initializer=initializer,
              name='%s/all' % self.name,
              reuse=tf.AUTO_REUSE) for v_i in embeddings[:-1]
      ]
      p = [
          self.func(v_dot[i], embeddings[j])
          for i, j in itertools.combinations(range(field_size), 2)
      ]
    elif self.bilinear_type == 'each':
      v_dot = [
          tf.layers.dense(
              v_i,
              dim,
              kernel_initializer=initializer,
              name='%s/each_%d' % (self.name, i),
              reuse=tf.AUTO_REUSE) for i, v_i in enumerate(embeddings[:-1])
      ]
      p = [
          self.func(v_dot[i], embeddings[j])
          for i, j in itertools.combinations(range(field_size), 2)
      ]
    else:  # interaction
      p = [
          self.func(
              tf.layers.dense(
                  embeddings[i],
                  embeddings[j].shape.as_list()[-1],
                  kernel_initializer=initializer,
                  name='%s/interaction_%d_%d' % (self.name, i, j),
                  reuse=tf.AUTO_REUSE), embeddings[j])
          for i, j in itertools.combinations(range(field_size), 2)
      ]

    output = tf.layers.dense(
        tf.concat(p, axis=-1), self.output_size, kernel_initializer=initializer)
    return output
