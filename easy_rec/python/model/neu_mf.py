# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.neu_mf_pb2 import NeuMF as NeuMFConfig  # NOQA

if tf.__version__ >= '2.0':
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
  tf = tf.compat.v1
else:
  losses = tf.losses
  metrics = tf.metrics


class NeuMF(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(NeuMF, self).__init__(model_config, feature_configs, features, labels,
                                is_training)
    assert self._model_config.WhichOneof('model') == 'neu_mf', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.neu_mf
    assert isinstance(self._model_config, NeuMFConfig)

    self.user_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.mlp_user_feature = self.user_feature
    self.mf_user_feature = self.user_feature
    self.item_feature, _ = self._input_layer(self._feature_dict, 'item')
    self.mlp_item_feature = self.item_feature
    self.mf_item_feature = self.item_feature
    # self.mlp_user_feature, _ = self._input_layer(self._feature_dict, 'mlp_user')
    # self.mf_user_feature, _ = self._input_layer(self._feature_dict, 'mf_user')
    # self.mlp_item_feature, _ = self._input_layer(self._feature_dict, 'mlp_item')
    # self.mf_item_feature, _ = self._input_layer(self._feature_dict, 'mf_item')

    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def build_predict_graph(self):
    # gmf_output = self.mf_user_feature * self.mf_item_feature
    gmf_output = tf.multiply(self.mf_user_feature, self.mf_item_feature)

    MLP = tf.concat([self.mlp_user_feature, self.mlp_item_feature], -1)

    dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
                        self._is_training)
    mlp_outputs = dnn_layer(MLP)
    all_fea = tf.concat([gmf_output, mlp_outputs], axis=-1)
    output = tf.layers.dense(all_fea, 1, name='output')

    self._add_to_prediction_dict(output)

    return self._prediction_dict

  def get_outputs(self):
    outputs = super(NeuMF, self).get_outputs()
    return outputs
