# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras.blocks import MLP
from easy_rec.python.layers.keras.layer_norm import LayerNormalization
from easy_rec.python.layers.utils import Parameter


class MaskBlock(Layer):
  """MaskBlock use in MaskNet.

  Args:
    projection_dim: project dimension to reduce the computational cost.
    Default is `None` such that a full (`input_dim` by `aggregation_size`) matrix
    W is used. If enabled, a low-rank matrix W = U*V will be used, where U
    is of size `input_dim` by `projection_dim` and V is of size
    `projection_dim` by `aggregation_size`. `projection_dim` need to be smaller
    than `aggregation_size`/2 to improve the model efficiency. In practice, we've
    observed that `projection_dim` = d/4 consistently preserved the
    accuracy of a full-rank version.
  """

  def __init__(self, params, name='mask_block', reuse=None, **kwargs):
    super(MaskBlock, self).__init__(name=name, **kwargs)
    self.config = params.get_pb_config()
    self.l2_reg = params.l2_regularizer
    self._projection_dim = params.get_or_default('projection_dim', None)
    self.reuse = reuse
    self.final_relu = Activation('relu', name='relu')

  def build(self, input_shape):
    if type(input_shape) in (tuple, list):
      assert len(input_shape) >= 2, 'MaskBlock must has at least two inputs'
      input_dim = int(input_shape[0][-1])
      mask_input_dim = int(input_shape[1][-1])
    else:
      input_dim, mask_input_dim = input_shape[-1], input_shape[-1]
    if self.config.HasField('reduction_factor'):
      aggregation_size = int(mask_input_dim * self.config.reduction_factor)
    elif self.config.HasField('aggregation_size') is not None:
      aggregation_size = self.config.aggregation_size
    else:
      raise ValueError(
          'Need one of reduction factor or aggregation size for MaskBlock.')

    self.aggr_layer = Dense(
        aggregation_size,
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=self.l2_reg,
        name='aggregation')
    self.weight_layer = Dense(input_dim, name='weights')
    if self._projection_dim is not None:
      logging.info('%s project dim is %d', self.name, self._projection_dim)
      self.project_layer = Dense(
          self._projection_dim,
          kernel_regularizer=self.l2_reg,
          use_bias=False,
          name='project')
    if self.config.input_layer_norm:
      # 推荐在调用MaskBlock之前做好 layer norm，否则每一次调用都需要对input做ln
      if tf.__version__ >= '2.0':
        self.input_layer_norm = tf.keras.layers.LayerNormalization(
            name='input_ln')
      else:
        self.input_layer_norm = LayerNormalization(name='input_ln')

    if self.config.HasField('output_size'):
      self.output_layer = Dense(
          self.config.output_size, use_bias=False, name='output')
    if tf.__version__ >= '2.0':
      self.output_layer_norm = tf.keras.layers.LayerNormalization(
          name='output_ln')
    else:
      self.output_layer_norm = LayerNormalization(name='output_ln')
    super(MaskBlock, self).build(input_shape)

  def call(self, inputs, training=None, **kwargs):
    if type(inputs) in (tuple, list):
      net, mask_input = inputs[:2]
    else:
      net, mask_input = inputs, inputs

    if self.config.input_layer_norm:
      net = self.input_layer_norm(net)

    if self._projection_dim is None:
      aggr = self.aggr_layer(mask_input)
    else:
      u = self.project_layer(mask_input)
      aggr = self.aggr_layer(u)

    weights = self.weight_layer(aggr)
    masked_net = net * weights

    if not self.config.HasField('output_size'):
      return masked_net

    hidden = self.output_layer(masked_net)
    ln_hidden = self.output_layer_norm(hidden)
    return self.final_relu(ln_hidden)


class MaskNet(Layer):
  """MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask.

  Refer: https://arxiv.org/pdf/2102.07619.pdf
  """

  def __init__(self, params, name='mask_net', reuse=None, **kwargs):
    super(MaskNet, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self.params = params
    self.config = params.get_pb_config()
    if self.config.HasField('mlp'):
      p = Parameter.make_from_pb(self.config.mlp)
      p.l2_regularizer = params.l2_regularizer
      self.mlp = MLP(p, name='mlp', reuse=reuse)
    else:
      self.mlp = None

    self.mask_layers = []
    for i, block_conf in enumerate(self.config.mask_blocks):
      params = Parameter.make_from_pb(block_conf)
      params.l2_regularizer = self.params.l2_regularizer
      mask_layer = MaskBlock(params, name='block_%d' % i, reuse=self.reuse)
      self.mask_layers.append(mask_layer)

    if self.config.input_layer_norm:
      if tf.__version__ >= '2.0':
        self.input_layer_norm = tf.keras.layers.LayerNormalization(
            name='input_ln')
      else:
        self.input_layer_norm = LayerNormalization(name='input_ln')

  def call(self, inputs, training=None, **kwargs):
    if self.config.input_layer_norm:
      inputs = self.input_layer_norm(inputs)

    if self.config.use_parallel:
      mask_outputs = [
          mask_layer((inputs, inputs)) for mask_layer in self.mask_layers
      ]
      all_mask_outputs = tf.concat(mask_outputs, axis=1)
      if self.mlp is not None:
        output = self.mlp(all_mask_outputs, training=training)
      else:
        output = all_mask_outputs
      return output
    else:
      net = inputs
      for i, _ in enumerate(self.config.mask_blocks):
        mask_layer = self.mask_layers[i]
        net = mask_layer((net, inputs))

      if self.mlp is not None:
        output = self.mlp(net, training=training)
      else:
        output = net
      return output
