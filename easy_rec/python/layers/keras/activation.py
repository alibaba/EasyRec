# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Layer

import easy_rec.python.utils.activation

try:
  from tensorflow.python.ops.init_ops import Zeros
except ImportError:
  from tensorflow.python.ops.init_ops_v2 import Zeros

try:
  from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
  BatchNormalization = tf.keras.layers.BatchNormalization

try:
  unicode
except NameError:
  unicode = str


class Dice(Layer):
  """The Data Adaptive Activation Function in DIN.

  which can be viewed as a generalization of PReLu
  and can adaptively adjust the rectified point according to distribution of input data.

  Input shape
    - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis)
     when using this layer as the first layer in a model.

  Output shape
    - Same shape as the input.

  Arguments
    - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).
    - **epsilon** : Small float added to variance to avoid dividing by zero.

  References
    - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]
      https://arxiv.org/pdf/1706.06978.pdf
  """

  def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
    self.axis = axis
    self.epsilon = epsilon
    super(Dice, self).__init__(**kwargs)

  def build(self, input_shape):
    self.bn = BatchNormalization(
        axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
    self.alphas = self.add_weight(
        shape=(input_shape[-1],),
        initializer=Zeros(),
        dtype=tf.float32,
        name='dice_alpha')  # name='alpha_'+self.name
    super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
    self.uses_learning_phase = True

  def call(self, inputs, training=None, **kwargs):
    inputs_normed = self.bn(inputs, training=training)
    # tf.layers.batch_normalization(
    # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
    x_p = tf.sigmoid(inputs_normed)
    return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

  def compute_output_shape(self, input_shape):
    return input_shape

  @property
  def updates(self):
    return self.bn.updates

  def get_config(self,):
    config = {'axis': self.axis, 'epsilon': self.epsilon}
    base_config = super(Dice, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MaskedSoftmax(Layer):

  def __init__(self, axis=-1, **kwargs):
    super(MaskedSoftmax, self).__init__(**kwargs)
    self.axis = axis

  def call(self, inputs, mask=None):
    if mask is not None:
      adder = (1.0 - tf.cast(mask, inputs.dtype)) * -1e9
      inputs += adder
    # Calculate softmax
    if isinstance(self.axis, (tuple, list)):
      if len(self.axis) > 1:
        raise ValueError('MaskedSoftmax not support multiple axis')
      else:
        return tf.nn.softmax(inputs, axis=self.axis[0])
    return tf.nn.softmax(inputs, axis=self.axis)


def activation_layer(activation, name=None):
  if activation in ('dice', 'Dice'):
    act_layer = Dice(name=name)
  elif isinstance(activation, (str, unicode)):
    act_fn = easy_rec.python.utils.activation.get_activation(activation)
    act_layer = Activation(act_fn, name=name)
  elif issubclass(activation, Layer):
    act_layer = activation(name=name)
  else:
    raise ValueError(
        'Invalid activation,found %s.You should use a str or a Activation Layer Class.'
        % (activation))
  return act_layer
