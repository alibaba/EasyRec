# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import tensorflow as tf
from easy_rec.python.compat.array_ops import repeat
if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class NumericalEmbedding(object):

  def __init__(self, embedding_dim, scope='numerical_embedding', stddev=1.0):
    self.embedding_dim = embedding_dim
    self.scope = scope
    self.initializer = tf.random_normal_initializer(stddev=stddev)

  def __call__(self, inputs, *args, **kwargs):
    if inputs.shape.ndims != 2:
      raise ValueError('inputs of NumericalEmbedding must have 2 dimensions.')

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
