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
            init_gate_bias=-1.0,
            reuse=None):
  if isinstance(activation, six.string_types):
    activation = get_activation(activation)
  with tf.variable_scope(scope, reuse):
    if size is None:
      size = x.shape.as_list()[-1]
    else:
      x = tf.layers.dense(x, size, name='input_projection', reuse=reuse)

    initializer = tf.constant_initializer(init_gate_bias)
    for i in range(num_layers):
      T = tf.layers.dense(
          x,
          size,
          activation=tf.sigmoid,
          bias_initializer=initializer,
          name='gate_%d' % i,
          reuse=reuse)
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

  def __init__(self, input_layer, feature_dict, group_name, reuse=None):
    self._group_name = group_name
    self.name = 'input_' + self._group_name
    self._input_layer = input_layer
    self._feature_dict = feature_dict
    self._reuse = reuse
    self.built = False

  def __call__(self, config, is_training, **kwargs):
    if not self.built:
      self.build(config, is_training)

    if config.output_seq_and_normal_feature:
      return self.inputs

    if config.do_batch_norm and config.do_layer_norm:
      raise ValueError(
          'can not do batch norm and layer norm for input layer at the same time'
      )
    with tf.name_scope(self.name):
      return self.call(config, is_training)

  def build(self, config, training):
    self.built = True
    combine = not config.output_seq_and_normal_feature
    self.inputs = self._input_layer(
        self._feature_dict, self._group_name, is_combine=combine)
    if config.output_seq_and_normal_feature:
      seq_feature_and_len, _, target_features = self.inputs
      seq_len = seq_feature_and_len[0][1]
      seq_features = [seq_fea for seq_fea, _ in seq_feature_and_len]
      if config.concat_seq_feature:
        if target_features:
          target_features = tf.concat(target_features, axis=-1)
        else:
          target_features = None
        assert len(
            seq_features) > 0, '[%s] sequence feature is empty' % self.name
        seq_features = tf.concat(seq_features, axis=-1)
      self.inputs = seq_features, seq_len, target_features
    self.reset(config, training)

  def reset(self, config, training):
    if 0.0 < config.dropout_rate < 1.0:
      self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)

    if training and 0.0 < config.feature_dropout_rate < 1.0:
      keep_prob = 1.0 - config.feature_dropout_rate
      self.bern = tf.distributions.Bernoulli(probs=keep_prob, dtype=tf.float32)

  def call(self, config, training):
    features, feature_list = self.inputs
    num_features = len(feature_list)

    do_ln = config.do_layer_norm
    do_bn = config.do_batch_norm
    do_feature_dropout = training and 0.0 < config.feature_dropout_rate < 1.0
    if do_feature_dropout:
      keep_prob = 1.0 - config.feature_dropout_rate
      mask = self.bern.sample(num_features)
    elif do_bn:
      features = tf.layers.batch_normalization(
          features, training=training, reuse=self._reuse)
    elif do_ln:
      features = layer_norm(
          features, name=self._group_name + '_features', reuse=self._reuse)

    output_feature_list = config.output_2d_tensor_and_feature_list
    output_feature_list = output_feature_list or config.only_output_feature_list
    output_feature_list = output_feature_list or config.only_output_3d_tensor
    rate = config.dropout_rate
    do_dropout = 0.0 < rate < 1.0
    if do_feature_dropout or do_ln or do_bn or do_dropout:
      for i in range(num_features):
        fea = feature_list[i]
        if do_bn:
          fea = tf.layers.batch_normalization(
              fea, training=training, reuse=self._reuse)
        elif do_ln:
          ln_name = self._group_name + 'f_%d' % i
          fea = layer_norm(fea, name=ln_name, reuse=self._reuse)
        if do_dropout and output_feature_list:
          fea = self.dropout.call(fea, training=training)
        if do_feature_dropout:
          fea = tf.div(fea, keep_prob) * mask[i]
        feature_list[i] = fea
      if do_feature_dropout:
        features = tf.concat(feature_list, axis=-1)

    if do_dropout and not do_feature_dropout:
      features = self.dropout.call(features, training=training)
    if features.shape.ndims == 3 and int(features.shape[0]) == 1:
      features = tf.squeeze(features, axis=0)

    if config.only_output_feature_list:
      return feature_list
    if config.only_output_3d_tensor:
      return tf.stack(feature_list, axis=1)
    if config.output_2d_tensor_and_feature_list:
      return features, feature_list
    return features
