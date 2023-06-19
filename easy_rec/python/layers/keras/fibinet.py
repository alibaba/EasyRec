# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import logging

import tensorflow as tf

from easy_rec.python.layers.common_layers import layer_norm
from easy_rec.python.layers.keras.blocks import MLP
from easy_rec.python.layers.utils import Parameter

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SENet(tf.keras.layers.Layer):
  """SENET Layer used in FiBiNET.

  Input shape
    - A list of 2D tensor with shape: ``(batch_size,embedding_size)``.
      The ``embedding_size`` of each field can have different value.

  Output shape
    - A 2D tensor with shape: ``(batch_size,sum_of_embedding_size)``.

  References:
    1. [FiBiNET](https://arxiv.org/pdf/1905.09433.pdf)
      Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    2. [FiBiNet++](https://arxiv.org/pdf/2209.05016.pdf)
      Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction
  """

  def __init__(self, params, name='SENet', **kwargs):
    super(SENet, self).__init__(name, **kwargs)
    self.config = params.get_pb_config()

  def call(self, inputs, **kwargs):
    g = self.config.num_squeeze_group
    for emb in inputs:
      assert emb.shape.ndims == 2, 'field embeddings must be rank 2 tensors'
      dim = int(emb.shape[-1])
      assert dim >= g and dim % g == 0, 'field embedding dimension %d must be divisible by %d' % (
          dim, g)

    field_size = len(inputs)
    feature_size_list = [emb.shape.as_list()[-1] for emb in inputs]

    # Squeeze
    # embedding dimension 必须能被 g 整除
    group_embs = [
        tf.reshape(emb, [-1, g, int(emb.shape[-1]) // g]) for emb in inputs
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
    inputs = tf.concat(inputs, axis=-1)
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


class BiLinear(tf.keras.layers.Layer):
  """BilinearInteraction Layer used in FiBiNET.

  Input shape
    - A list of 2D tensor with shape: ``(batch_size,embedding_size)``.
      Its length is ``filed_size``.
      The ``embedding_size`` of each field can have different value.

  Output shape
    - 2D tensor with shape: ``(batch_size,output_size)``.

  Attributes:
    num_output_units: the number of output units
    type: ['all', 'each', 'interaction'], types of bilinear functions used in this layer
    use_plus: whether to use bi-linear+

  References:
    1. [FiBiNET](https://arxiv.org/pdf/1905.09433.pdf)
      Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    2. [FiBiNet++](https://arxiv.org/pdf/2209.05016.pdf)
      Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction
  """

  def __init__(self, params, name='bilinear', **kwargs):
    super(BiLinear, self).__init__(name, **kwargs)
    params.check_required(['num_output_units'])
    bilinear_plus = params.get_or_default('use_plus', True)
    self.bilinear_type = params.get_or_default('type', 'interaction').lower()
    self.output_size = params.num_output_units

    if self.bilinear_type not in ['all', 'each', 'interaction']:
      raise NotImplementedError(
          "bilinear_type only support: ['all', 'each', 'interaction']")

    if bilinear_plus:
      self.func = _full_interaction
    else:
      self.func = tf.multiply

  def call(self, inputs, **kwargs):
    embeddings = inputs
    logging.info('Bilinear Layer with %d inputs' % len(embeddings))
    if len(embeddings) > 200:
      logging.warning('There are too many inputs for bilinear layer: %d' %
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


class FiBiNet(tf.keras.layers.Layer):
  """FiBiNet++:Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction.

  References:
    - [FiBiNet++](https://arxiv.org/pdf/2209.05016.pdf)
      Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction
  """

  def __init__(self, params, name='fibinet', **kwargs):
    super(FiBiNet, self).__init__(name, **kwargs)
    self._config = params.get_pb_config()
    if self._config.HasField('mlp'):
      p = Parameter.make_from_pb(self._config.mlp)
      p.l2_regularizer = params.l2_regularizer
      self.final_mlp = MLP(p, name=name)
    else:
      self.final_mlp = None

  def call(self, inputs, training=None, **kwargs):
    feature_list = []

    params = Parameter.make_from_pb(self._config.senet)
    senet = SENet(params, name='%s/senet' % self.name)
    senet_output = senet(inputs)
    feature_list.append(senet_output)

    if self._config.HasField('bilinear'):
      params = Parameter.make_from_pb(self._config.bilinear)
      bilinear = BiLinear(params, name='%s/bilinear' % self.name)
      bilinear_output = bilinear(inputs)
      feature_list.append(bilinear_output)

    if len(feature_list) > 1:
      feature = tf.concat(feature_list, axis=-1)
    else:
      feature = feature_list[0]

    if self.final_mlp is not None:
      feature = self.final_mlp(feature, training=training)
    return feature
