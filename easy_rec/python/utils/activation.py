# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from easy_rec.python.utils.load_class import load_by_path

try:
  from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
  BatchNormalization = tf.keras.layers.BatchNormalization


class Dice(Layer):
  """The Data Adaptive Activation Function in DIN.

  Which can be viewed as a generalization of PReLu, and can adaptively adjust the rectified point
   according to distribution of input data.

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
     Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
     ACM, 2018: 1059-1068.] (https://arxiv.org/pdf/1706.06978.pdf)
  """

  def __init__(self,
               feat_dim,
               axis=-1,
               epsilon=1e-9,
               is_training=None,
               **kwargs):
    super(Dice, self).__init__(**kwargs)
    self.feat_dim = feat_dim
    self.axis = axis
    self.epsilon = epsilon
    self.is_training = is_training
    self.bn = BatchNormalization(
        axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
    self.alphas = tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)

  def call(self, inputs, **kwargs):
    inputs_normed = self.bn(inputs, training=self.is_training)
    x_p = tf.sigmoid(inputs_normed)
    return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self,):
    config = {'axis': self.axis, 'epsilon': self.epsilon}
    base_config = super(Dice, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string, **kwargs):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """
  # We assume that anything that's not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == 'linear':
    return None
  elif act == 'relu':
    return tf.nn.relu
  elif act == 'gelu':
    return gelu
  elif act == 'leaky_relu':
    return tf.nn.leaky_relu
  elif act in ('prelu', 'PRelu'):
    if len(kwargs) == 0:
      return tf.nn.leaky_relu
    return tf.keras.layers.PReLU(**kwargs)
  elif act in ('dice', 'Dice'):
    return Dice(**kwargs)
  elif act == 'elu':
    return tf.nn.elu
  elif act == 'selu':
    return tf.nn.selu
  elif act == 'tanh':
    return tf.tanh
  elif act == 'swish':
    if tf.__version__ < '1.13.0':
      return lambda x: x * tf.sigmoid(x)
    return tf.nn.swish
  elif act == 'sigmoid':
    return tf.nn.sigmoid
  else:
    return load_by_path(activation_string)
