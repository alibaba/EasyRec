# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict
import math
import numpy as np
import tensorflow as tf

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

    # complexity = self._config.feature_complexity_weight * config.complexity
    #
    # # dim = 1.0
    # # if config.HasField('embedding_dim'):
    # #   dim = float(config.embedding_dim)
    # dim = self.features_dimension[name]
    # complexity += self._config.feature_dimension_weight * dim
    #
    # cardinal = 1.0
    # if config.HasField('hash_bucket_size'):
    #   cardinal = float(config.hash_bucket_size)
    # elif config.HasField('num_buckets'):
    #   cardinal = float(config.num_buckets)
    # elif len(config.boundaries) > 0:
    #   cardinal = float(len(config.boundaries) + 1)
    # complexity += self._config.feature_cardinality_weight * cardinal
    #
    # theta = 1.0 - sigmoid(complexity)
    # alpha = math.log(1.0 - theta) - math.log(theta)
    # feature_regularize[name] = alpha

  return feature_complexity


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

  def compute_dropout_mask(self, n, temperature=0.1):
    delta_name = 'delta' if self.name == 'all' else 'delta_%s' % self.name
    delta = tf.get_variable(
        name=delta_name,
        shape=[n],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.5))

    EPSILON = np.finfo(float).eps
    unif_noise = tf.random_uniform([n], dtype=tf.float32, seed=None, name='uniform_noise')

    approx = (
        tf.log(delta + EPSILON) - tf.log(1. - delta + EPSILON) +
        tf.log(unif_noise + EPSILON) - tf.log(1. - unif_noise + EPSILON))
    return tf.sigmoid(approx / temperature)

  def compute_regular_params(self, cols_to_feature):
    alphas = OrderedDict()
    for fc, fea in cols_to_feature.items():
      dim = int(fea.shape[-1])
      complexity = self.feature_complexity[fc.raw_name]
      cardinal = 1
      if isinstance(fc, EmbeddingColumn) or isinstance(fc, SharedEmbeddingColumn):
        cardinal = fc.cardinality
      c = self._config.feature_complexity_weight * complexity
      c += self._config.feature_cardinality_weight * cardinal
      c += self._config.feature_dimension_weight * dim
      theta = 1.0 - sigmoid(complexity)
      alpha = math.log(1.0 - theta) - math.log(theta)
      alphas[fc] = alpha
    return alphas

  def __call__(self, cols_to_feature):
    """
    cols_to_feature: an ordered dict mapping feature_column to feature_values
    """
    output_tensors = []
    alphas = []
    z = self.compute_dropout_mask(len(cols_to_feature))  # keep ratio
    regular = self.compute_regular_params(cols_to_feature)
    feature_columns = cols_to_feature.keys()
    for column in sorted(feature_columns, key=lambda x: x.name):
      value = cols_to_feature[column]
      alpha = regular[column]
      i = len(output_tensors)
      out = value * z[i] if self.is_training else value
      cols_to_feature[column] = out
      output_tensors.append(out)
      alphas.append(alpha)

    output_features = tf.concat(output_tensors, 1)

    batch_size = tf.shape(output_features)[0]
    t_alpha = tf.convert_to_tensor(alphas)  # [M]
    loss = tf.reduce_sum(t_alpha * z) / batch_size

    tf.add_to_collection('variational_dropout_loss', loss)
    return output_features


def sigmoid(x):
  return 1. / (1. + math.exp(-x))
