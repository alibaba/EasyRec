# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import six
import tensorflow as tf

from easy_rec.python.compat.layers import layer_norm as tf_layer_norm
from easy_rec.python.utils.activation import get_activation

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def highway(x,
            size=None,
            activation=None,
            num_layers=1,
            scope='highway',
            dropout=0.0,
            reuse=None):
  if isinstance(activation, six.string_types):
    activation = get_activation(activation)
  with tf.variable_scope(scope, reuse):
    if size is None:
      size = x.shape.as_list()[-1]
    else:
      x = tf.layers.dense(x, size, name='input_projection', reuse=reuse)

    for i in range(num_layers):
      T = tf.layers.dense(
          x, size, activation=tf.sigmoid, name='gate_%d' % i, reuse=reuse)
      H = tf.layers.dense(
          x, size, activation=activation, name='activation_%d' % i, reuse=reuse)
      if dropout > 0.0:
        H = tf.nn.dropout(H, 1.0 - dropout)
      x = H * T + x * (1.0 - T)
    return x


def text_cnn(x,
             filter_sizes=(3, 4, 5),
             num_filters=(128, 64, 64),
             scope_name='textcnn',
             reuse=False):
  # x: None * step_dim * embed_dim
  assert len(filter_sizes) == len(num_filters)
  initializer = tf.variance_scaling_initializer()
  pooled_outputs = []
  for i in range(len(filter_sizes)):
    filter_size = filter_sizes[i]
    num_filter = num_filters[i]
    scope_name_i = scope_name + '_' + str(filter_size)
    with tf.variable_scope(scope_name_i, reuse=reuse):
      # conv shape: (batch_size, seq_len - filter_size + 1, num_filters)
      conv = tf.layers.conv1d(
          x,
          filters=int(num_filter),
          kernel_size=int(filter_size),
          activation=tf.nn.relu,
          name='conv_layer',
          reuse=reuse,
          kernel_initializer=initializer,
          padding='same')
      pool = tf.reduce_max(
          conv, axis=1)  # max pooling, shape: (batch_size, num_filters)
    pooled_outputs.append(pool)
  pool_flat = tf.concat(
      pooled_outputs, 1)  # shape: (batch_size, num_filters * len(filter_sizes))
  return pool_flat


def layer_norm(input_tensor, name=None, reuse=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf_layer_norm(
      inputs=input_tensor,
      begin_norm_axis=-1,
      begin_params_axis=-1,
      reuse=reuse,
      scope=name)


class EnhancedInputLayer(object):
  """Enhance the raw input layer."""

  def __init__(self, config, input_layer, feature_dict):
    if config.do_batch_norm and config.do_layer_norm:
      raise ValueError(
          'can not do batch norm and layer norm for input layer at the same time'
      )
    self._config = config
    self._input_layer = input_layer
    self._feature_dict = feature_dict

  def __call__(self, group, is_training, **kwargs):
    with tf.name_scope('input_' + group):
      return self.call(group, is_training)

  def call(self, group, is_training):
    if self._config.output_seq_and_normal_feature:
      seq_features, target_feature, target_features = self._input_layer(
          self._feature_dict, group, is_combine=False)
      return seq_features, target_features

    features, feature_list = self._input_layer(self._feature_dict, group)
    num_features = len(feature_list)

    do_ln = self._config.do_layer_norm
    do_bn = self._config.do_batch_norm
    do_feature_dropout = is_training and 0.0 < self._config.feature_dropout_rate < 1.0
    if do_feature_dropout:
      keep_prob = 1.0 - self._config.feature_dropout_rate
      bern = tf.distributions.Bernoulli(probs=keep_prob)
      mask = bern.sample(num_features)
    elif do_bn:
      features = tf.layers.batch_normalization(features, training=is_training)
    elif do_ln:
      features = layer_norm(features)

    do_dropout = 0.0 < self._config.dropout_rate < 1.0
    if do_feature_dropout or do_ln or do_bn or do_dropout:
      for i in range(num_features):
        fea = feature_list[i]
        if self._config.do_batch_norm:
          fea = tf.layers.batch_normalization(fea, training=is_training)
        elif self._config.do_layer_norm:
          fea = layer_norm(fea)
        if do_dropout:
          fea = tf.layers.dropout(
              fea, self._config.dropout_rate, training=is_training)
        if do_feature_dropout:
          fea = tf.div(fea, keep_prob) * mask[i]
        feature_list[i] = fea
      if do_feature_dropout:
        features = tf.concat(feature_list, axis=-1)

    if do_dropout and not do_feature_dropout:
      features = tf.layers.dropout(
          features, self._config.dropout_rate, training=is_training)

    if self._config.only_output_feature_list:
      return feature_list
    if self._config.only_output_3d_tensor:
      return tf.stack(feature_list, axis=1)
    if self._config.output_2d_tensor_and_feature_list:
      return features, feature_list
    return features


class Concatenate(object):

  def __init__(self, config):
    self.config = config

  def __call__(self, inputs, *args, **kwargs):
    if self.config.HasField('expand_dim_before'):
      dim = self.config.expand_dim_before
      output = tf.stack(inputs, axis=dim)
    else:
      output = tf.concat(inputs, axis=self.config.axis)

    if self.config.HasField('expand_dim_after'):
      dim = self.config.expand_dim_after
      output = tf.expand_dims(output, dim)
    return output
