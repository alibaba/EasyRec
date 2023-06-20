# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.multi_tower_recall_pb2 import MultiTowerRecall as MultiTowerRecallConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MultiTowerRecall(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MultiTowerRecall, self).__init__(model_config, feature_configs,
                                           features, labels, is_training)
    assert self._model_config.WhichOneof('model') == 'multi_tower_recall', (
        'invalid model config: %s' % self._model_config.WhichOneof('model'))
    self._model_config = self._model_config.multi_tower_recall
    assert isinstance(self._model_config, MultiTowerRecallConfig)

    self.user_tower_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.item_tower_feature, _ = self._input_layer(self._feature_dict, 'item')

  def build_predict_graph(self):

    user_tower_feature = self.user_tower_feature
    batch_size = tf.shape(user_tower_feature)[0]
    pos_item_feature = self.item_tower_feature[:batch_size]
    neg_item_feature = self.item_tower_feature[batch_size:]
    item_tower_feature = tf.concat([
        pos_item_feature[:, tf.newaxis, :],
        tf.tile(
            neg_item_feature[tf.newaxis, :, :], multiples=[batch_size, 1, 1])
    ],
                                   axis=1)  # noqa: E126

    user_dnn = dnn.DNN(self._model_config.user_tower.dnn, self._l2_reg,
                       'user_dnn', self._is_training)
    user_tower_emb = user_dnn(user_tower_feature)

    item_dnn = dnn.DNN(self._model_config.item_tower.dnn, self._l2_reg,
                       'item_dnn', self._is_training)
    item_tower_emb = item_dnn(item_tower_feature)
    item_tower_emb = tf.reshape(item_tower_emb, tf.shape(user_tower_emb))

    tower_fea_arr = []
    tower_fea_arr.append(user_tower_emb)
    tower_fea_arr.append(item_tower_emb)

    all_fea = tf.concat(tower_fea_arr, axis=-1)
    final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                              'final_dnn', self._is_training)
    all_fea = final_dnn_layer(all_fea)
    output = tf.layers.dense(all_fea, 1, name='output')
    output = output[:, 0]

    self._add_to_prediction_dict(output)

    return self._prediction_dict
