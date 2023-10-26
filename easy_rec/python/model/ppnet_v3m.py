# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

import tensorflow as tf
from tensorflow.python.framework import ops

from easy_rec.python.layers import dnn
from easy_rec.python.layers import mmoe
from easy_rec.python.model.keep_model.model_ps_mmoe import CustomizedModel
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos.ppnet_pb2 import PPNet as PPNetConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class PPNetV3M(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(PPNetV3M, self).__init__(model_config, feature_configs, features,
                                   labels, is_training)
    assert self._model_config.WhichOneof('model') == 'ppnet', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.ppnet
    assert isinstance(self._model_config, PPNetConfig)

    self._features, _ = self._input_layer(self._feature_dict, 'all')

    with open(self._model_config.model_conf, 'r') as fin:
      self._model_conf = json.load(fin)
    indim = self._features.get_shape()[1]
    logging.info('indim = %d' % indim)
    self._keras_model = CustomizedModel(self._model_conf, indim)

  def build_predict_graph(self):
    # self._add_to_prediction_dict(tower_outputs)
    output_list = self._keras_model(self._features, self._is_training)
    trainable_variables = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    for var in self._keras_model.trainable_variables:
      if var not in trainable_variables:
        ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, var)

    # update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    # for var in self._keras_model.updates:
    #   if var not in update_ops:
    #     ops.add_to_collection(ops.GraphKeys.UPDATE_OPS, var)

    for lbl_id in range(len(self._model_conf['label'])):
      lbl_info = self._model_conf['label'][lbl_id]
      lbl_name = lbl_info.get('input_name')
      output = output_list[lbl_id]
      self._prediction_dict[lbl_name] = output
    # return self._prediction_dict

  def build_loss_graph(self):
    for lbl_id in range(len(self._model_conf['label'])):
      lbl_info = self._model_conf['label'][lbl_id]
      lbl_name = lbl_info.get('input_name')
      output = self._prediction_dict.get(lbl_name)
      loss_obj = tf.keras.losses.BinaryCrossentropy(reduction='sum')(
          self._labels[lbl_name], output)
      self._loss_dict[lbl_name] = loss_obj
    return self._loss_dict
