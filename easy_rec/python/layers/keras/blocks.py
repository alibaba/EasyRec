# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for building models."""
import logging

import tensorflow as tf
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras.activation import activation_layer
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.utils.shape_utils import pad_or_truncate_sequence
from easy_rec.python.utils.tf_utils import add_elements_to_collection

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MLP(Layer):
  """Sequential multi-layer perceptron (MLP) block.

  Attributes:
    units: Sequential list of layer sizes.
    use_bias: Whether to include a bias term.
    activation: Type of activation to use on all except the last layer.
    final_activation: Type of activation to use on last layer.
    **kwargs: Extra args passed to the Keras Layer base class.
  """

  def __init__(self, params, name='mlp', reuse=None, **kwargs):
    super(MLP, self).__init__(name=name, **kwargs)
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
    for i, num_units in enumerate(units[:-1]):
      name = 'layer_%d' % i
      drop_rate = dropout_rate[i] if i < num_dropout else 0.0
      self.add_rich_layer(num_units, use_bn, drop_rate, activation, initializer,
                          use_bias, use_bn_after_act, name,
                          params.l2_regularizer)

    n = len(units) - 1
    drop_rate = dropout_rate[n] if num_dropout > n else 0.0
    name = 'layer_%d' % n
    self.add_rich_layer(units[-1], use_final_bn, drop_rate, final_activation,
                        initializer, use_final_bias, use_bn_after_act, name,
                        params.l2_regularizer)

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
    act_layer = activation_layer(activation)
    if use_bn and not use_bn_after_activation:
      dense = Dense(
          units=num_units,
          use_bias=use_bias,
          kernel_initializer=initializer,
          kernel_regularizer=l2_reg,
          name=name)
      self._sub_layers.append(dense)
      bn = tf.keras.layers.BatchNormalization(
          name='%s/bn' % name, trainable=True)
      self._sub_layers.append(bn)
      self._sub_layers.append(act_layer)
    else:
      dense = Dense(
          num_units,
          use_bias=use_bias,
          kernel_initializer=initializer,
          kernel_regularizer=l2_reg,
          name=name)
      self._sub_layers.append(dense)
      self._sub_layers.append(act_layer)
      if use_bn and use_bn_after_activation:
        bn = tf.keras.layers.BatchNormalization(name='%s/bn' % name)
        self._sub_layers.append(bn)

    if 0.0 < dropout_rate < 1.0:
      dropout = Dropout(dropout_rate, name='%s/dropout' % name)
      self._sub_layers.append(dropout)
    elif dropout_rate >= 1.0:
      raise ValueError('invalid dropout_ratio: %.3f' % dropout_rate)

  def call(self, x, training=None, **kwargs):
    """Performs the forward computation of the block."""
    for layer in self._sub_layers:
      cls = layer.__class__.__name__
      if cls in ('Dropout', 'BatchNormalization', 'Dice'):
        x = layer(x, training=training)
        if cls in ('BatchNormalization', 'Dice'):
          add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
      else:
        x = layer(x)
    return x


class Highway(Layer):

  def __init__(self, params, name='highway', reuse=None, **kwargs):
    super(Highway, self).__init__(name, **kwargs)
    self.emb_size = params.get_or_default('emb_size', None)
    self.num_layers = params.get_or_default('num_layers', 1)
    self.activation = params.get_or_default('activation', 'relu')
    self.dropout_rate = params.get_or_default('dropout_rate', 0.0)
    self.init_gate_bias = params.get_or_default('init_gate_bias', -3.0)
    self.act_layer = activation_layer(self.activation)
    self.dropout_layer = Dropout(
        self.dropout_rate) if self.dropout_rate > 0.0 else None
    self.project_layer = None
    self.gate_bias_initializer = Constant(self.init_gate_bias)
    self.gates = []  # T
    self.transforms = []  # H
    self.multiply_layer = tf.keras.layers.Multiply()
    self.add_layer = tf.keras.layers.Add()

  def build(self, input_shape):
    dim = input_shape[-1]
    if self.emb_size is not None and dim != self.emb_size:
      self.project_layer = Dense(self.emb_size, name='input_projection')
      dim = self.emb_size
    self.carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))
    for i in range(self.num_layers):
      gate = Dense(
          units=dim,
          bias_initializer=self.gate_bias_initializer,
          activation='sigmoid',
          name='gate_%d' % i)
      self.gates.append(gate)
      self.transforms.append(Dense(units=dim))

  def call(self, inputs, training=None, **kwargs):
    value = inputs
    if self.project_layer is not None:
      value = self.project_layer(inputs)
    for i in range(self.num_layers):
      gate = self.gates[i](value)
      transformed = self.act_layer(self.transforms[i](value))
      if self.dropout_layer is not None:
        transformed = self.dropout_layer(transformed, training=training)
      transformed_gated = self.multiply_layer([gate, transformed])
      identity_gated = self.multiply_layer([self.carry_gate(gate), value])
      value = self.add_layer([transformed_gated, identity_gated])
    return value


class Gate(Layer):
  """Weighted sum gate."""

  def __init__(self, params, name='gate', reuse=None, **kwargs):
    super(Gate, self).__init__(name, **kwargs)
    self.weight_index = params.get_or_default('weight_index', 0)

  def call(self, inputs, **kwargs):
    assert len(
        inputs
    ) > 1, 'input of Gate layer must be a list containing at least 2 elements'
    weights = inputs[self.weight_index]
    j = 0
    for i, x in enumerate(inputs):
      if i == self.weight_index:
        continue
      if j == 0:
        output = weights[:, j, None] * x
      else:
        output += weights[:, j, None] * x
      j += 1
    return output


class TextCNN(Layer):
  """Text CNN Model.

  References
  - [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
  """

  def __init__(self, params, name='text_cnn', reuse=None, **kwargs):
    super(TextCNN, self).__init__(name, **kwargs)
    self.config = params.get_pb_config()
    self.pad_seq_length = self.config.pad_sequence_length
    if self.pad_seq_length <= 0:
      logging.warning(
          'run text cnn with pad_sequence_length <= 0, the predict of model may be unstable'
      )
    self.conv_layers = []
    self.pool_layer = tf.keras.layers.GlobalMaxPool1D()
    self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
    for size, filters in zip(self.config.filter_sizes, self.config.num_filters):
      conv = tf.keras.layers.Conv1D(
          filters=int(filters),
          kernel_size=int(size),
          activation=self.config.activation)
      self.conv_layers.append(conv)
    if self.config.HasField('mlp'):
      p = Parameter.make_from_pb(self.config.mlp)
      p.l2_regularizer = params.l2_regularizer
      self.mlp = MLP(p, name='mlp', reuse=reuse)
    else:
      self.mlp = None

  def call(self, inputs, training=None, **kwargs):
    """Input shape: 3D tensor with shape: `(batch_size, steps, input_dim)."""
    assert isinstance(inputs, (list, tuple))
    assert len(inputs) >= 2
    seq_emb, seq_len = inputs[:2]

    if self.pad_seq_length > 0:
      seq_emb, seq_len = pad_or_truncate_sequence(seq_emb, seq_len,
                                                  self.pad_seq_length)
    pooled_outputs = []
    for layer in self.conv_layers:
      conv = layer(seq_emb)
      pooled = self.pool_layer(conv)
      pooled_outputs.append(pooled)
    net = self.concat_layer(pooled_outputs)
    if self.mlp is not None:
      output = self.mlp(net)
    else:
      output = net
    return output
