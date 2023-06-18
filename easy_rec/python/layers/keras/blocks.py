# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for building models."""
import logging

import tensorflow as tf

from easy_rec.python.utils.activation import get_activation


class MLP(tf.keras.layers.Layer):
  """Sequential multi-layer perceptron (MLP) block.

  Attributes:
    units: Sequential list of layer sizes.
    use_bias: Whether to include a bias term.
    activation: Type of activation to use on all except the last layer.
    final_activation: Type of activation to use on last layer.
    **kwargs: Extra args passed to the Keras Layer base class.
  """

  def __init__(self, params, name='mlp', **kwargs):
    super(MLP, self).__init__(name=name, **kwargs)
    params.check_required('hidden_units')
    use_bn = params.get_or_default('use_bn', True)
    use_final_bn = params.get_or_default('use_final_bn', True)
    use_bias = params.get_or_default('use_bias', True)
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

    num_dropout = len(dropout_rate)
    self._sub_layers = []
    for i, num_units in enumerate(units[:-1]):
      name = 'dnn_%d' % i
      drop_rate = dropout_rate[i] if i < num_dropout else 0.0
      self.add_rich_layer(num_units, use_bn, drop_rate, activation, initializer,
                          use_bias, use_bn_after_act, name,
                          params.l2_regularizer)

    n = len(units) - 1
    drop_rate = dropout_rate[n] if num_dropout > n else 0.0
    name = 'dnn_%d' % n
    self.add_rich_layer(units[-1], use_final_bn, drop_rate, final_activation,
                        initializer, use_bias, use_bn_after_act, name,
                        params.l2_regularizer)

  def add_rich_layer(self,
                     num_units,
                     use_bn,
                     dropout_rate,
                     activation,
                     initializer,
                     use_bias=True,
                     use_bn_after_activation=False,
                     name='mlp',
                     l2_reg=None):
    act_fn = get_activation(activation)
    if use_bn and not use_bn_after_activation:
      dense = tf.keras.layers.Dense(
          units=num_units,
          use_bias=use_bias,
          kernel_initializer=initializer,
          kernel_regularizer=l2_reg,
          name=name)
      self._sub_layers.append(dense)
      # bn = tf.keras.layers.BatchNormalization(name='%s/bn' % name)
      # keras BN layer have a stale issue on some versions of tf
      bn = lambda x, training: tf.layers.batch_normalization(
          x, training=training, name='%s/%s/bn' % (self.name, name))
      self._sub_layers.append(bn)
      act = tf.keras.layers.Activation(act_fn, name='%s/act' % name)
      self._sub_layers.append(act)
    else:
      dense = tf.keras.layers.Dense(
          num_units,
          activation=act_fn,
          use_bias=use_bias,
          kernel_initializer=initializer,
          kernel_regularizer=l2_reg,
          name=name)
      self._sub_layers.append(dense)
      if use_bn and use_bn_after_activation:
        bn = lambda x, training: tf.layers.batch_normalization(
            x, training=training, name='%s/%s/bn' % (self.name, name))
        self._sub_layers.append(bn)

    if 0.0 < dropout_rate < 1.0:
      dropout = tf.keras.layers.Dropout(dropout_rate, name='%s/dropout' % name)
      self._sub_layers.append(dropout)
    elif dropout_rate >= 1.0:
      raise ValueError('invalid dropout_ratio: %.3f' % dropout_rate)

  def call(self, x, training=None, **kwargs):
    """Performs the forward computation of the block."""
    for layer in self._sub_layers:
      x = layer(x, training=training)
    return x


class Highway(tf.keras.layers.Layer):

  def __init__(self, params, name='highway', **kwargs):
    super(Highway, self).__init__(name, **kwargs)
    params.check_required('emb_size')
    self.emb_size = params.emb_size
    self.num_layers = params.get_or_default('num_layers', 1)
    self.activation = params.get_or_default('activation', 'gelu')
    self.dropout_rate = params.get_or_default('dropout_rate', 0.0)

  def call(self, inputs, training=None, **kwargs):
    from easy_rec.python.layers.common_layers import highway
    return highway(
        inputs,
        self.emb_size,
        activation=self.activation,
        num_layers=self.num_layers,
        dropout=self.dropout_rate if training else 0.0)
