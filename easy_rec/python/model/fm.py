# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import tensorflow as tf

from easy_rec.python.layers import fm
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos.fm_pb2 import FM as FMConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class FM(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(FM, self).__init__(model_config, feature_configs, features, labels,
                             is_training)
    assert self._model_config.WhichOneof('model') == 'fm', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.fm
    assert isinstance(self._model_config, FMConfig)

    self._wide_features, _ = self._input_layer(self._feature_dict, 'wide')
    self._deep_features, self._fm_features = self._input_layer(
        self._feature_dict, 'deep')

  def build_input_layer(self, model_config, feature_configs):
    # overwrite create input_layer to support wide_output_dim
    self._wide_output_dim = model_config.num_class
    super(FM, self).build_input_layer(model_config, feature_configs)

  def build_predict_graph(self):
    wide_fea = tf.reduce_sum(
        self._wide_features, axis=1, keepdims=True, name='wide_feature')

    fm_fea = fm.FM(name='fm_feature')(self._fm_features)

    if self._num_class > 1:
      fm_fea = tf.layers.dense(
          fm_fea,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='fm_logits')
    else:
      fm_fea = tf.reduce_sum(fm_fea, 1, keepdims=True)

    bias = tf.get_variable(
        'fm_bias', [self._num_class],
        initializer=tf.zeros_initializer(),
        trainable=True)

    output = wide_fea + fm_fea
    output = tf.nn.bias_add(output, bias)

    self._add_to_prediction_dict(output)
    return self._prediction_dict
