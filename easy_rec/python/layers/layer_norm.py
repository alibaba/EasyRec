# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class LayerNormalization(tf.layers.Layer):
  """Layer normalization for BTC format: supports L2(default) and L1 modes."""

  def __init__(self, hidden_size, params={}):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.norm_type = params.get('type', 'layernorm_L2')
    self.epsilon = params.get('epsilon', 1e-6)

  def build(self, _):
    self.scale = tf.get_variable(
        'layer_norm_scale', [self.hidden_size],
        initializer=tf.keras.initializers.Ones(),
        dtype=tf.float32)
    self.bias = tf.get_variable(
        'layer_norm_bias', [self.hidden_size],
        initializer=tf.keras.initializers.Zeros(),
        dtype=tf.float32)
    self.built = True

  def call(self, x):
    if self.norm_type == 'layernorm_L2':
      epsilon = self.epsilon
      dtype = x.dtype
      x = tf.cast(x=x, dtype=tf.float32)
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
      norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
      result = norm_x * self.scale + self.bias
      return tf.cast(x=result, dtype=dtype)

    else:
      dtype = x.dtype
      if dtype == tf.float16:
        x = tf.cast(x, dtype=tf.float32)
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      x = x - mean
      variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
      norm_x = tf.div(x, variance + self.epsilon)
      y = norm_x * self.scale + self.bias
      if dtype == tf.float16:
        y = tf.saturate_cast(y, dtype)
      return y
