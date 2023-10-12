# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
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
    _, self._deep_features = self._input_layer(self._feature_dict, 'deep')

  def build_input_layer(self, model_config, feature_configs):
    # overwrite create input_layer to support wide_output_dim
    has_final = len(model_config.wide_and_deep.final_dnn.hidden_units) > 0
    self._wide_output_dim = model_config.wide_and_deep.wide_output_dim
    if not has_final:
      model_config.wide_and_deep.wide_output_dim = model_config.num_class
      self._wide_output_dim = model_config.num_class
    super(WideAndDeep, self).build_input_layer(model_config, feature_configs)

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

  def get_grouped_vars(self, opt_num):
    """Group the vars into different optimization groups.

    Each group will be optimized by a separate optimizer.

    Args:
      opt_num: number of optimizers from easyrec config.

    Return:
      list of list of variables.
    """
    assert opt_num <= 3, 'could only support 2 or 3 optimizers, ' + \
        'if opt_num = 2, one for the wide , and one for the others, ' + \
        'if opt_num = 3, one for the wide, second for the deep embeddings, ' + \
        'and third for the other layers.'

    if opt_num == 2:
      wide_vars = []
      deep_vars = []
      for tmp_var in tf.trainable_variables():
        if tmp_var.name.startswith('input_layer') and \
            (not tmp_var.name.startswith('input_layer_1')):
          wide_vars.append(tmp_var)
        else:
          deep_vars.append(tmp_var)
      return [wide_vars, deep_vars]
    elif opt_num == 3:
      wide_vars = []
      embedding_vars = []
      deep_vars = []
      for tmp_var in tf.trainable_variables():
        if tmp_var.name.startswith('input_layer') and \
            (not tmp_var.name.startswith('input_layer_1')):
          wide_vars.append(tmp_var)
        elif tmp_var.name.startswith(
            'input_layer') or '/embedding_weights' in tmp_var.name:
          embedding_vars.append(tmp_var)
        else:
          deep_vars.append(tmp_var)
      return [wide_vars, embedding_vars, deep_vars]
