# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json

import numpy as np
import tensorflow as tf

from easy_rec.python.compat.feature_column.feature_column import _SharedEmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import EmbeddingColumn  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class VariationalDropoutLayer(object):
  """Rank features by variational dropout.

  Use the Dropout concept on the input feature layer and optimize the corresponding feature-wise dropout rate
  paper: Dropout Feature Ranking for Deep Learning Models
  arXiv: 1712.08645
  """

  def __init__(self,
               variational_dropout_config,
               features_dimension,
               is_training=False,
               name=''):
    self._config = variational_dropout_config
    self.features_dimension = features_dimension
    self.features_total_dimension = sum(self.features_dimension.values())

    if self.variational_dropout_wise():
      self._dropout_param_size = self.features_total_dimension
      self.drop_param_shape = [self._dropout_param_size]
    else:
      self._dropout_param_size = len(self.features_dimension)
      self.drop_param_shape = [self._dropout_param_size]
    self.evaluate = not is_training

    logit_p_name = 'logit_p' if name == 'all' else 'logit_p_%s' % name
    self.logit_p = tf.get_variable(
        name=logit_p_name,
        shape=self.drop_param_shape,
        dtype=tf.float32,
        initializer=None)
    tf.add_to_collection(
        'variational_dropout',
        json.dumps([name, list(self.features_dimension.items())]))

  def get_lambda(self):
    return self._config.regularization_lambda

  def variational_dropout_wise(self):
    return self._config.embedding_wise_variational_dropout

  def build_expand_index(self, batch_size):
    # Build index_list--->[[0,0],[0,0],[0,0],[0,0],[0,1]......]
    expanded_index = []
    for i, index_loop_count in enumerate(self.features_dimension.values()):
      for m in range(index_loop_count):
        expanded_index.append([i])
    expanded_index = tf.tile(expanded_index, [batch_size, 1])
    batch_size_range = tf.range(batch_size)
    expand_range_axis = tf.expand_dims(batch_size_range, 1)
    batch_size_range_expand_dim_len = tf.tile(
        expand_range_axis, [1, self.features_total_dimension])
    index_i = tf.reshape(batch_size_range_expand_dim_len, [-1, 1])
    expanded_index = tf.concat([index_i, expanded_index], 1)
    return expanded_index

  def sample_noisy_input(self, input):
    batch_size = tf.shape(input)[0]
    if self.evaluate:
      expanded_dims_logit_p = tf.expand_dims(self.logit_p, 0)
      expanded_logit_p = tf.tile(expanded_dims_logit_p, [batch_size, 1])
      p = tf.sigmoid(expanded_logit_p)
      if self.variational_dropout_wise():
        scaled_input = input * (1 - p)
      else:
        # expand dropout layer
        expanded_index = self.build_expand_index(batch_size)
        expanded_p = tf.gather_nd(p, expanded_index)
        expanded_p = tf.reshape(expanded_p, [-1, self.features_total_dimension])
        scaled_input = input * (1 - expanded_p)

      return scaled_input
    else:
      bern_val = self.sampled_from_logit_p(batch_size)
      bern_val = tf.reshape(bern_val, [-1, self.features_total_dimension])
      noisy_input = input * bern_val
      return noisy_input

  def sampled_from_logit_p(self, num_samples):
    expand_dims_logit_p = tf.expand_dims(self.logit_p, 0)
    expand_logit_p = tf.tile(expand_dims_logit_p, [num_samples, 1])
    dropout_p = tf.sigmoid(expand_logit_p)
    bern_val = self.concrete_dropout_neuron(dropout_p)

    if self.variational_dropout_wise():
      return bern_val
    else:
      # from feature_num to embedding_dim_num
      expanded_index = self.build_expand_index(num_samples)
      bern_val_gather_nd = tf.gather_nd(bern_val, expanded_index)
      return bern_val_gather_nd

  def concrete_dropout_neuron(self, dropout_p, temp=1.0 / 10.0):
    EPSILON = np.finfo(float).eps
    unif_noise = tf.random_uniform(
        tf.shape(dropout_p), dtype=tf.float32, seed=None, name='unif_noise')

    approx = (
        tf.log(dropout_p + EPSILON) - tf.log(1. - dropout_p + EPSILON) +
        tf.log(unif_noise + EPSILON) - tf.log(1. - unif_noise + EPSILON))

    approx_output = tf.sigmoid(approx / temp)
    return 1 - approx_output

  def __call__(self, output_features):
    batch_size = tf.shape(output_features)[0]
    noisy_input = self.sample_noisy_input(output_features)
    dropout_p = tf.sigmoid(self.logit_p)
    variational_dropout_penalty = 1. - dropout_p
    variational_dropout_penalty_lambda = self.get_lambda() / tf.cast(
        batch_size, dtype=tf.float32)
    variational_dropout_loss_sum = variational_dropout_penalty_lambda * tf.reduce_sum(
        variational_dropout_penalty, axis=0)
    tf.add_to_collection('variational_dropout_loss',
                         variational_dropout_loss_sum)
    return noisy_input
