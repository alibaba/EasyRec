# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import six
import tensorflow as tf

from easy_rec.python.utils.load_class import load_by_path

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def dice(_x, axis=-1, epsilon=1e-9, name='dice', training=True):
  """The Data Adaptive Activation Function in DIN.

  Which can be viewed as a generalization of PReLu,
  and can adaptively adjust the rectified point according to distribution of input data.

  Arguments
    - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).
    - **epsilon** : Small float added to variance to avoid dividing by zero.

  References
    - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]
     Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
     ACM, 2018: 1059-1068.] (https://arxiv.org/pdf/1706.06978.pdf)
  """
  alphas = tf.get_variable(
      'alpha_' + name,
      _x.get_shape()[-1],
      initializer=tf.constant_initializer(0.0),
      dtype=tf.float32)
  inputs_normed = tf.layers.batch_normalization(
      inputs=_x,
      axis=axis,
      epsilon=epsilon,
      center=False,
      scale=False,
      training=training)
  x_p = tf.sigmoid(inputs_normed)
  return alphas * (1.0 - x_p) * _x + x_p * _x


def gelu(x, name='gelu'):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    x: float Tensor to perform activation.
    name: name for this activation

  Returns:
    `x` with the GELU activation applied.
  """
  with tf.name_scope(name):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x, name='swish'):
  with tf.name_scope(name):
    return x * tf.sigmoid(x)


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
  elif act == 'prelu':
    if len(kwargs) == 0:
      return tf.nn.leaky_relu
    return tf.keras.layers.PReLU(**kwargs)
  elif act == 'dice':
    return lambda x, name='dice': dice(x, name=name, **kwargs)
  elif act == 'elu':
    return tf.nn.elu
  elif act == 'selu':
    return tf.nn.selu
  elif act == 'tanh':
    return tf.tanh
  elif act == 'swish':
    if tf.__version__ < '1.13.0':
      return swish
    return tf.nn.swish
  elif act == 'sigmoid':
    return tf.nn.sigmoid
  else:
    return load_by_path(activation_string)
