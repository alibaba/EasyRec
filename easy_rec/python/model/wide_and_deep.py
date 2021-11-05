# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import input_layer
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.wide_and_deep_pb2 import WideAndDeep as WideAndDeepConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class WideAndDeep(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(WideAndDeep, self).__init__(model_config, feature_configs, features,
                                      labels, is_training)
    assert model_config.WhichOneof('model') == 'wide_and_deep', \
        'invalid model config: %s' % model_config.WhichOneof('model')
    self._model_config = model_config.wide_and_deep
    assert isinstance(self._model_config, WideAndDeepConfig)
    assert self._input_layer.has_group('wide')
    _, self._wide_features = self._input_layer(self._feature_dict, 'wide')
    assert self._input_layer.has_group('deep')
    _, self._deep_features = self._input_layer(
        self._feature_dict, exclude_group_names=['wide'])

  def build_input_layer(self, model_config, feature_configs):
    # overwrite create input_layer to support wide_output_dim
    has_final = len(model_config.wide_and_deep.final_dnn.hidden_units) > 0
    wide_output_dim = model_config.wide_and_deep.wide_output_dim
    if not has_final:
      model_config.wide_and_deep.wide_output_dim = model_config.num_class
      wide_output_dim = model_config.num_class
    self._input_layer = input_layer.InputLayer(
        feature_configs,
        model_config.feature_groups,
        seq_feature_groups_config=model_config.seq_att_groups,
        wide_output_dim=wide_output_dim,
        use_embedding_variable=model_config.use_embedding_variable,
        embedding_regularizer=self._emb_reg,
        kernel_regularizer=self._l2_reg)

  def build_predict_graph(self):
    wide_fea = tf.add_n(self._wide_features)
    logging.info('wide features dimension: %d' % wide_fea.get_shape()[-1])

    self._deep_features = tf.concat(self._deep_features, axis=1)
    logging.info('input deep features dimension: %d' %
                 self._deep_features.get_shape()[-1])

    deep_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'deep_feature',
                         self._is_training)
    deep_fea = deep_layer(self._deep_features)
    logging.info('output deep features dimension: %d' %
                 deep_fea.get_shape()[-1])

    has_final = len(self._model_config.final_dnn.hidden_units) > 0
    print('wide_deep has_final_dnn layers = %d' % has_final)
    if has_final:
      all_fea = tf.concat([wide_fea, deep_fea], axis=1)
      final_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                            'final_dnn', self._is_training)
      all_fea = final_layer(all_fea)
      output = tf.layers.dense(
          all_fea,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='output')
    else:
      deep_out = tf.layers.dense(
          deep_fea,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='deep_out')
      output = deep_out + wide_fea

    self._add_to_prediction_dict(output)

    return self._prediction_dict
