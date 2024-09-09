# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import logging

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras.blocks import MLP
from easy_rec.python.layers.keras.layer_norm import LayerNormalization
from easy_rec.python.layers.utils import Parameter


class SENet(Layer):
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

  def __init__(self, params, name='SENet', reuse=None, **kwargs):
    super(SENet, self).__init__(name=name, **kwargs)
    self.config = params.get_pb_config()
    self.reuse = reuse
    if tf.__version__ >= '2.0':
      self.layer_norm = tf.keras.layers.LayerNormalization(name='output_ln')
    else:
      self.layer_norm = LayerNormalization(name='output_ln')

  def build(self, input_shape):
    g = self.config.num_squeeze_group
    emb_size = 0
    for shape in input_shape:
      assert shape.ndims == 2, 'field embeddings must be rank 2 tensors'
      dim = int(shape[-1])
      assert dim >= g and dim % g == 0, 'field embedding dimension %d must be divisible by %d' % (
          dim, g)
      emb_size += dim

    r = self.config.reduction_ratio
    field_size = len(input_shape)
    reduction_size = max(1, field_size * g * 2 // r)
    self.reduce_layer = Dense(
        units=reduction_size,
        activation='relu',
        kernel_initializer='he_normal',
        name='W1')
    self.excite_layer = Dense(
        units=emb_size, kernel_initializer='glorot_normal', name='W2')
    super(SENet, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, **kwargs):
    g = self.config.num_squeeze_group

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
    a1 = self.reduce_layer(z)
    weights = self.excite_layer(a1)

    # Re-weight
    inputs = tf.concat(inputs, axis=-1)
    output = inputs * weights

    # Fuse, add skip-connection
    if self.config.use_skip_connection:
      output += inputs

    # Layer Normalization
    if self.config.use_output_layer_norm:
      output = self.layer_norm(output)
    return output


def _full_interaction(v_i, v_j):
  # [bs, 1, dim] x [bs, dim, 1] = [bs, 1]
  interaction = tf.matmul(
      tf.expand_dims(v_i, axis=1), tf.expand_dims(v_j, axis=-1))
  return tf.squeeze(interaction, axis=1)


class BiLinear(Layer):
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

  def __init__(self, params, name='bilinear', reuse=None, **kwargs):
    super(BiLinear, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    params.check_required(['num_output_units'])
    bilinear_plus = params.get_or_default('use_plus', True)
    self.output_size = params.num_output_units
    self.bilinear_type = params.get_or_default('type', 'interaction').lower()
    if self.bilinear_type not in ['all', 'each', 'interaction']:
      raise NotImplementedError(
          "bilinear_type only support: ['all', 'each', 'interaction']")
    if bilinear_plus:
      self.func = _full_interaction
    else:
      self.func = tf.multiply
    self.output_layer = Dense(self.output_size, name='output')

  def build(self, input_shape):
    if type(input_shape) not in (tuple, list):
      raise TypeError('input of BiLinear layer must be a list')
    field_num = len(input_shape)
    logging.info('Bilinear Layer with %d inputs' % field_num)
    if field_num > 200:
      logging.warning('Too many inputs for bilinear layer: %d' % field_num)
    equal_dim = True
    _dim = input_shape[0][-1]
    for shape in input_shape:
      assert shape.ndims == 2, 'field embeddings must be rank 2 tensors'
      if shape[-1] != _dim:
        equal_dim = False
    if not equal_dim and self.bilinear_type != 'interaction':
      raise ValueError(
          'all embedding dimensions must be same when not use bilinear type: interaction'
      )
    dim = int(_dim)

    if self.bilinear_type == 'all':
      self.dot_layer = Dense(dim, name='all')
    elif self.bilinear_type == 'each':
      self.dot_layers = [
          Dense(dim, name='each_%d' % i) for i in range(field_num - 1)
      ]
    else:  # interaction
      self.dot_layers = [
          Dense(
              units=int(input_shape[j][-1]), name='interaction_%d_%d' % (i, j))
          for i, j in itertools.combinations(range(field_num), 2)
      ]
    super(BiLinear, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, **kwargs):
    embeddings = inputs
    field_num = len(embeddings)

    # bi-linear+: dimension of `p` is [bs, f*(f-1)/2]
    # bi-linear:
    #   - when equal_dim=True, dimension of `p` is [bs, f*(f-1)/2*k], k is embedding size
    #   - when equal_dim=False, dimension of `p` is [bs, (k_2+k_3+...+k_f)+...+(k_i+k_{i+1}+...+k_f)+...+k_f],
    #   - where k_i is the embedding size of the ith field
    if self.bilinear_type == 'all':
      v_dot = [self.dot_layer(v_i) for v_i in embeddings[:-1]]
      p = [
          self.func(v_dot[i], embeddings[j])
          for i, j in itertools.combinations(range(field_num), 2)
      ]
    elif self.bilinear_type == 'each':
      v_dot = [self.dot_layers[i](v_i) for i, v_i in enumerate(embeddings[:-1])]
      p = [
          self.func(v_dot[i], embeddings[j])
          for i, j in itertools.combinations(range(field_num), 2)
      ]
    else:  # interaction
      p = [
          self.func(self.dot_layers[i * field_num + j](embeddings[i]),
                    embeddings[j])
          for i, j in itertools.combinations(range(field_num), 2)
      ]

    return self.output_layer(tf.concat(p, axis=-1))


class FiBiNet(Layer):
  """FiBiNet++:Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction.

  References:
    - [FiBiNet++](https://arxiv.org/pdf/2209.05016.pdf)
      Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction
  """

  def __init__(self, params, name='fibinet', reuse=None, **kwargs):
    super(FiBiNet, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self._config = params.get_pb_config()

    se_params = Parameter.make_from_pb(self._config.senet)
    self.senet_layer = SENet(
        se_params, name=self.name + '/senet', reuse=self.reuse)

    if self._config.HasField('bilinear'):
      bi_params = Parameter.make_from_pb(self._config.bilinear)
      self.bilinear_layer = BiLinear(
          bi_params, name=self.name + '/bilinear', reuse=self.reuse)

    if self._config.HasField('mlp'):
      p = Parameter.make_from_pb(self._config.mlp)
      p.l2_regularizer = params.l2_regularizer
      self.final_mlp = MLP(p, name=self.name + '/mlp', reuse=reuse)
    else:
      self.final_mlp = None

  def call(self, inputs, training=None, **kwargs):
    feature_list = []

    senet_output = self.senet_layer(inputs)
    feature_list.append(senet_output)

    if self._config.HasField('bilinear'):
      feature_list.append(self.bilinear_layer(inputs))

    if len(feature_list) > 1:
      feature = tf.concat(feature_list, axis=-1)
    else:
      feature = feature_list[0]

    if self.final_mlp is not None:
      feature = self.final_mlp(feature, training=training)
    return feature
