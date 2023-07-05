# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import math
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.meta_graph import read_meta_graph_file

from easy_rec.python.compat.sort_ops import argsort

from easy_rec.python.compat.feature_column.feature_column import _SharedEmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import EmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import SharedEmbeddingColumn  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def get_feature_complexity(feature_configs):
  feature_complexity = {}
  for config in feature_configs:
    name = config.input_names[0]
    if config.HasField('feature_name'):
      name = config.feature_name
    feature_complexity[name] = config.complexity
  return feature_complexity


def sigmoid(x):
  return 1. / (1. + math.exp(-x))


def get_feature_importance(pipeline_config, feature_group_name=None):
  assert pipeline_config.model_config.HasField(
      'variational_dropout'), 'variational_dropout must be in model_config'

  checkpoint_path = tf.train.latest_checkpoint(pipeline_config.model_dir)
  meta_graph_def = read_meta_graph_file(checkpoint_path + '.meta')

  features_map = dict()
  for col_def in meta_graph_def.collection_def[
      'variational_dropout'].bytes_list.value:
    features = json.loads(col_def)
    features_map.update(features)

  feature_importance = OrderedDict()
  tf.logging.info('Reading checkpoint from %s ...' % checkpoint_path)
  reader = tf.train.NewCheckpointReader(checkpoint_path)
  for feature_group in pipeline_config.model_config.feature_groups:
    group_name = feature_group.group_name
    if feature_group_name is not None and feature_group_name != group_name:
      continue
    # assert group_name in features_map, "%s not in feature map" % group_name
    if group_name not in features_map:
      # for now, sequence feature groups are not supported
      logging.warn('%s not in feature map' % group_name)
      continue

    feature_dims = features_map[group_name]

    delta_name = 'fscd_delta_%s' % group_name
    if not reader.has_tensor(delta_name):
      logging.warn("feature group `%s` doesn't be involved in FSCD layer")
      for feature, dim in feature_dims:
        feature_importance[feature] = 1.0
      continue

    delta = reader.get_tensor(delta_name)
    indices = argsort(delta, direction='DESCENDING')
    keep_prob = tf.nn.sigmoid(delta)
    with tf.Session() as sess:
      idx = indices.eval(session=sess)
      probs = keep_prob.eval(session=sess)
    for i in idx:
      feature = feature_dims[i][0]
      if feature in feature_importance:
        raw = feature_importance[feature]
        if probs[i] > raw:
          logging.info('%s importance change from %d to %d', feature, raw,
                       probs[i])
          feature_importance[feature] = probs[i]
      else:
        feature_importance[feature] = probs[i]
  return feature_importance


class FSCDLayer(object):
  """Rank features by variational dropout.

  paper: Towards a Better Tradeoff between Effectiveness and Efficiency in Pre-Ranking,
    A Learnable Feature Selection based Approach
  arXiv: 2105.07706
  """

  def __init__(self,
               feature_configs,
               variational_dropout_config,
               is_training=False,
               name=''):
    self._config = variational_dropout_config
    self.is_training = is_training
    self.name = name
    self.feature_complexity = get_feature_complexity(feature_configs)

  def compute_dropout_mask(self, n):
    delta_name = 'fscd_delta_%s' % self.name
    delta = tf.get_variable(
        name=delta_name,
        shape=[n],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    delta = tf.nn.sigmoid(delta)
    epsilon = np.finfo(float).eps
    max_keep_ratio = self._config.max_keep_ratio
    min_keep_ratio = self._config.min_keep_ratio
    if max_keep_ratio >= 1.0:
      max_keep_ratio = 1.0 - epsilon
    if min_keep_ratio <= 0.0:
      min_keep_ratio = epsilon
    delta = tf.clip_by_value(delta, min_keep_ratio, max_keep_ratio)

    unif_noise = tf.random_uniform([n],
                                   dtype=tf.float32,
                                   seed=None,
                                   name='uniform_noise')
    approx = (
        tf.log(delta) - tf.log(1. - delta) + tf.log(unif_noise) -
        tf.log(1. - unif_noise))
    return tf.sigmoid(approx / self._config.temperature), delta

  def compute_regular_params(self, cols_to_feature):
    alphas = {}
    for fc, fea in cols_to_feature.items():
      dim = int(fea.shape[-1])
      complexity = self.feature_complexity[fc.raw_name]
      cardinal = 1
      if isinstance(fc, EmbeddingColumn) or isinstance(
          fc, _SharedEmbeddingColumn) or isinstance(fc, SharedEmbeddingColumn):
        cardinal = fc.cardinality
      c = self._config.feature_complexity_weight * complexity
      c += self._config.feature_cardinality_weight * cardinal
      c += self._config.feature_dimension_weight * dim
      sig_c = sigmoid(c)
      theta = 1.0 - sig_c
      alpha = math.log(sig_c) - math.log(theta)
      alphas[fc] = alpha
      print(
          str(fc.raw_name), 'complexity:', complexity, 'cardinality:', cardinal,
          'dimension:', dim, 'c:', c, 'theta:', theta, 'alpha:', alpha)
    return alphas

  def __call__(self, cols_to_feature):
    """cols_to_feature: an ordered dict mapping feature_column to feature_values."""
    feature_dimension = []
    output_tensors = []
    alphas = []
    z, delta = self.compute_dropout_mask(len(cols_to_feature))  # keep ratio
    tf.summary.histogram('fscd_keep_ratio', delta)
    tf.summary.histogram('fscd_keep_mask', z)
    regular = self.compute_regular_params(cols_to_feature)

    feature_columns = cols_to_feature.keys()
    for column in sorted(feature_columns, key=lambda x: x.name):
      value = cols_to_feature[column]
      alpha = regular[column]
      i = len(output_tensors)
      if self.is_training:
        scaled_value = tf.div(value, delta[i])
        out = tf.multiply(scaled_value, z[i], name='fscd_dropout')
      else:
        out = value
      cols_to_feature[column] = out
      output_tensors.append(out)
      alphas.append(alpha)
      feature_dimension.append((column.raw_name, int(value.shape[-1])))

    output_features = tf.concat(output_tensors, 1)
    tf.add_to_collection('variational_dropout',
                         json.dumps({self.name: feature_dimension}))

    batch_size = tf.shape(output_features)[0]
    t_alpha = tf.convert_to_tensor(alphas, dtype=tf.float32)
    loss = tf.reduce_sum(t_alpha * z) / tf.to_float(batch_size)

    tf.add_to_collection('variational_dropout_loss', loss)
    return output_features


# def dropout(p):
#    u = np.random.uniform()
#    x = math.log(p) - math.log(1-p) + math.log(u) - math.log(1-u)
#    z = sigmoid(x/0.1)
#    return z
#
#
# if __name__ == '__main__':
#    for i in range(100):
#      print(dropout(0.5))
