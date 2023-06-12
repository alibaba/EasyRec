# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import tensorflow as tf
from easy_rec.python.compat.array_ops import repeat
if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class PeriodicEmbedding(object):

  def __init__(self, embedding_dim, scope='periodic_embedding', stddev=1.0):
    """On Embeddings for Numerical Features in Tabular Deep Learning.

    Refer: https://arxiv.org/pdf/2203.05556.pdf
    """
    self.embedding_dim = embedding_dim // 2
    self.scope = scope
    self.initializer = tf.random_normal_initializer(stddev=stddev)

  def __call__(self, inputs, *args, **kwargs):
    if inputs.shape.ndims != 2:
      raise ValueError('inputs of PeriodicEmbedding must have 2 dimensions.')

    num_features = int(inputs.shape[-1])
    with tf.variable_scope(self.scope):
      c = tf.get_variable(
          'coef',
          shape=[1, num_features * self.embedding_dim],
          initializer=self.initializer)

      features = repeat(inputs, self.embedding_dim, axis=1)
      v = features * c * 2 * math.pi
      sin_v = tf.split(tf.sin(v), num_features, axis=1)
      cos_v = tf.split(tf.cos(v), num_features, axis=1)

      embeddings = []
      for val in zip(sin_v, cos_v):
        embedding = tf.concat(val, axis=1)
        embedding = tf.layers.dense(embedding, int(embedding.shape[-1]), activation=tf.nn.relu)
        embeddings.append(embedding)
      return tf.concat(embeddings, axis=1)


class AutoDisEmbedding(object):
  def __init__(self, config, scope='auto_dis'):
    """An Embedding Learning Framework for Numerical Features in CTR Prediction.

    Refer: https://arxiv.org/pdf/2012.08986v2.pdf
    """
    self.config = config
    self.emb_dim = config.embedding_dim
    self.num_bins = config.num_bins
    self.scope = scope

  def __call__(self, inputs, *args, **kwargs):
    if inputs.shape.ndims != 2:
      raise ValueError('inputs of PeriodicEmbedding must have 2 dimensions.')

    num_features = int(inputs.shape[-1])
    with tf.variable_scope(self.scope):
      meta_emb = tf.get_variable('meta_embedding', shape=[1, num_features, self.num_bins, self.emb_dim])
      w = tf.get_variable('project_w', shape=[1, num_features, self.num_bins])
      mat = tf.get_variable('project_mat', shape=[1, num_features, self.num_bins, self.num_bins])

      x = tf.expand_dims(inputs, axis=-1)  # [B, num_fea, 1]
      hidden = tf.nn.leaky_relu(w * x)  # [B, num_fea, num_bin]

      y = tf.matmul(mat, tf.expand_dims(hidden, axis=-1))  # [B, num_fea, num_bin, 1]
      y = tf.squeeze(y, axis=3)  # [B, num_fea, num_bin]

      # keep_prob(float): if dropout_flag is True, keep_prob rate to keep connect; (float, keep_prob=0.8)
      alpha = self.config.keep_prob
      x_bar = y + alpha * hidden  # [B, num_fea, num_bin]
      x_hat = tf.nn.softmax(x_bar / self.config.temperature)  # [B, num_fea, num_bin]

      emb = tf.matmul(tf.expand_dims(x_hat, axis=2), meta_emb)  # [B, num_fea, 1, emb_dim]
      # emb = tf.squeeze(emb, axis=2)  # [B, num_fea, emb_dim]
      return tf.reshape(emb, [-1, self.emb_dim * num_features])  # [B, num_fea*emb_dim]
