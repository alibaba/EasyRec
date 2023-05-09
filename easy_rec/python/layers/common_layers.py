# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def highway(x,
            size=None,
            activation=None,
            num_layers=1,
            scope='highway',
            dropout=0.0,
            reuse=None):
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
