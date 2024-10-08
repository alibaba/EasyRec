# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for building models."""
import logging

import tensorflow as tf

from easy_rec.python.layers.keras.activation import activation_layer
from easy_rec.python.utils.tf_utils import add_elements_to_collection

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class GateNN(tf.keras.layers.Layer):

  def __init__(self,
               params,
               output_units=None,
               name='gate_nn',
               reuse=None,
               **kwargs):
    super(GateNN, self).__init__(name=name, **kwargs)
    output_dim = output_units if output_units is not None else params.output_dim
    hidden_dim = params.get_or_default('hidden_dim', output_dim)
    initializer = params.get_or_default('initializer', 'he_uniform')
    do_batch_norm = params.get_or_default('use_bn', False)
    activation = params.get_or_default('activation', 'relu')
    dropout_rate = params.get_or_default('dropout_rate', 0.0)

    self._sub_layers = []
    dense = tf.keras.layers.Dense(
        units=hidden_dim,
        use_bias=not do_batch_norm,
        kernel_initializer=initializer)
    self._sub_layers.append(dense)

    if do_batch_norm:
      bn = tf.keras.layers.BatchNormalization(trainable=True)
      self._sub_layers.append(bn)

    act_layer = activation_layer(activation)
    self._sub_layers.append(act_layer)

    if 0.0 < dropout_rate < 1.0:
      dropout = tf.keras.layers.Dropout(dropout_rate)
      self._sub_layers.append(dropout)
    elif dropout_rate >= 1.0:
      raise ValueError('invalid dropout_ratio: %.3f' % dropout_rate)

    dense = tf.keras.layers.Dense(
        units=output_dim,
        activation='sigmoid',
        use_bias=not do_batch_norm,
        kernel_initializer=initializer,
        name='weight')
    self._sub_layers.append(dense)
    self._sub_layers.append(lambda x: x * 2)

  def call(self, x, training=None, **kwargs):
    """Performs the forward computation of the block."""
    for layer in self._sub_layers:
      cls = layer.__class__.__name__
      if cls in ('Dropout', 'BatchNormalization', 'Dice'):
        x = layer(x, training=training)
        if cls in ('BatchNormalization', 'Dice') and training:
          add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
      else:
        x = layer(x)
    return x


class PPNet(tf.keras.layers.Layer):
  """PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information.

  Attributes:
    units: Sequential list of layer sizes.
    use_bias: Whether to include a bias term.
    activation: Type of activation to use on all except the last layer.
    final_activation: Type of activation to use on last layer.
    **kwargs: Extra args passed to the Keras Layer base class.
  """

  def __init__(self, params, name='ppnet', reuse=None, **kwargs):
    super(PPNet, self).__init__(name=name, **kwargs)
    params.check_required('mlp')
    self.full_gate_input = params.get_or_default('full_gate_input', True)
    mode = params.get_or_default('mode', 'lazy')
    gate_params = params.gate_params
    params = params.mlp
    params.check_required('hidden_units')
    use_bn = params.get_or_default('use_bn', True)
    use_final_bn = params.get_or_default('use_final_bn', True)
    use_bias = params.get_or_default('use_bias', False)
    use_final_bias = params.get_or_default('use_final_bias', False)
    dropout_rate = list(params.get_or_default('dropout_ratio', []))
    activation = params.get_or_default('activation', 'relu')
    initializer = params.get_or_default('initializer', 'he_uniform')
    final_activation = params.get_or_default('final_activation', None)
    use_bn_after_act = params.get_or_default('use_bn_after_activation', False)
    units = list(params.hidden_units)
    logging.info(
        'MLP(%s) units: %s, dropout: %r, activate=%s, use_bn=%r, final_bn=%r,'
        ' final_activate=%s, bias=%r, initializer=%s, bn_after_activation=%r' %
        (name, units, dropout_rate, activation, use_bn, use_final_bn,
         final_activation, use_bias, initializer, use_bn_after_act))
    assert len(units) > 0, 'MLP(%s) takes at least one hidden units' % name
    self.reuse = reuse

    num_dropout = len(dropout_rate)
    self._sub_layers = []

    if mode != 'lazy':
      self._sub_layers.append(GateNN(gate_params, None, 'gate_0'))
    for i, num_units in enumerate(units[:-1]):
      name = 'layer_%d' % i
      drop_rate = dropout_rate[i] if i < num_dropout else 0.0
      self.add_rich_layer(num_units, use_bn, drop_rate, activation, initializer,
                          use_bias, use_bn_after_act, name,
                          params.l2_regularizer)
      self._sub_layers.append(
          GateNN(gate_params, num_units, 'gate_%d' % (i + 1)))

    n = len(units) - 1
    drop_rate = dropout_rate[n] if num_dropout > n else 0.0
    name = 'layer_%d' % n
    self.add_rich_layer(units[-1], use_final_bn, drop_rate, final_activation,
                        initializer, use_final_bias, use_bn_after_act, name,
                        params.l2_regularizer)
    if mode == 'lazy':
      self._sub_layers.append(
          GateNN(gate_params, units[-1], 'gate_%d' % (n + 1)))

  def add_rich_layer(self,
                     num_units,
                     use_bn,
                     dropout_rate,
                     activation,
                     initializer,
                     use_bias,
                     use_bn_after_activation,
                     name,
                     l2_reg=None):
    act_layer = activation_layer(activation, name='%s/act' % name)
    if use_bn and not use_bn_after_activation:
      dense = tf.keras.layers.Dense(
          units=num_units,
          use_bias=use_bias,
          kernel_initializer=initializer,
          kernel_regularizer=l2_reg,
          name='%s/dense' % name)
      self._sub_layers.append(dense)
      bn = tf.keras.layers.BatchNormalization(
          name='%s/bn' % name, trainable=True)
      self._sub_layers.append(bn)
      self._sub_layers.append(act_layer)
    else:
      dense = tf.keras.layers.Dense(
          num_units,
          use_bias=use_bias,
          kernel_initializer=initializer,
          kernel_regularizer=l2_reg,
          name='%s/dense' % name)
      self._sub_layers.append(dense)
      self._sub_layers.append(act_layer)
      if use_bn and use_bn_after_activation:
        bn = tf.keras.layers.BatchNormalization(name='%s/bn' % name)
        self._sub_layers.append(bn)

    if 0.0 < dropout_rate < 1.0:
      dropout = tf.keras.layers.Dropout(dropout_rate, name='%s/dropout' % name)
      self._sub_layers.append(dropout)
    elif dropout_rate >= 1.0:
      raise ValueError('invalid dropout_ratio: %.3f' % dropout_rate)

  def call(self, inputs, training=None, **kwargs):
    """Performs the forward computation of the block."""
    x, gate_input = inputs
    if self.full_gate_input:
      with tf.name_scope(self.name):
        gate_input = tf.concat([tf.stop_gradient(x), gate_input], axis=-1)

    for layer in self._sub_layers:
      cls = layer.__class__.__name__
      if cls == 'GateNN':
        gate = layer(gate_input)
        x *= gate
      elif cls in ('Dropout', 'BatchNormalization', 'Dice'):
        x = layer(x, training=training)
        if cls in ('BatchNormalization', 'Dice') and training:
          add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
      else:
        x = layer(x)
    return x
