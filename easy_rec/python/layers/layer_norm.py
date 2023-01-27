# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

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
    self.scale = variable_scope.get_variable(
        'layer_norm_scale', [self.hidden_size],
        initializer=tf.keras.initializers.Ones(),
        dtype=tf.float32)
    self.bias = variable_scope.get_variable(
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


def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None):
  """Adds a Layer Normalization layer.

  Based on the paper:

    "Layer Normalization"

    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    https://arxiv.org/abs/1607.06450.

  Can be used as a normalizer function for conv2d and fully_connected.

  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
  is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
  if requested, is performed over axes `begin_params_axis .. R - 1`.

  By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
  meaning that normalization is performed over all but the first axis
  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
  parameters are calculated for the rightmost axis (the `C` if `inputs` is
  `NHWC`).  Scaling and recentering is performed via broadcast of the
  `beta` and `gamma` parameters with the normalized tensor.

  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
  and this part of the inputs' shape must be fully defined.

  Args:
    inputs: A tensor having rank `R`. The normalization is performed over
      axes `begin_norm_axis ... R - 1` and centering and scaling parameters
      are calculated over `begin_params_axis ... R - 1`.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    begin_norm_axis: The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation, having the same
    shape and dtype as `inputs`.

  Raises:
    ValueError: If the rank of `inputs` is not known at graph build time,
      or if `inputs.shape[begin_params_axis:]` is not fully defined at
      graph build time.
  """
  with variable_scope.variable_scope(scope, 'LayerNorm', [inputs], reuse=reuse):
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                       'must be < rank(inputs) (%d)' %
                       (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
          (inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = variable_scope.get_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.zeros_initializer(),
          trainable=trainable)
    if scale:
      gamma = variable_scope.get_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1e-12
    outputs = nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs
