# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers import seq_input_layer
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.multi_tower_pb2 import MultiTower as MultiTowerConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MultiTowerDIN(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MultiTowerDIN, self).__init__(model_config, feature_configs, features,
                                        labels, is_training)
    self._seq_input_layer = seq_input_layer.SeqInputLayer(
        feature_configs,
        model_config.seq_att_groups,
        embedding_regularizer=self._emb_reg,
        ev_params=self._global_ev_params)
    assert self._model_config.WhichOneof('model') == 'multi_tower', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.multi_tower
    assert isinstance(self._model_config, MultiTowerConfig)

    self._tower_features = []
    self._tower_num = len(self._model_config.towers)
    for tower_id in range(self._tower_num):
      tower = self._model_config.towers[tower_id]
      tower_feature, _ = self._input_layer(self._feature_dict, tower.input)
      self._tower_features.append(tower_feature)

    self._din_tower_features = []
    self._din_tower_num = len(self._model_config.din_towers)

    logging.info('all tower num: {0}'.format(self._tower_num +
                                             self._din_tower_num))
    logging.info('din tower num: {0}'.format(self._din_tower_num))

    for tower_id in range(self._din_tower_num):
      tower = self._model_config.din_towers[tower_id]
      tower_feature = self._seq_input_layer(self._feature_dict, tower.input)

      # apply regularization for sequence feature key in seq_input_layer.

      regularizers.apply_regularization(
          self._emb_reg, weights_list=[tower_feature['hist_seq_emb']])
      self._din_tower_features.append(tower_feature)

  def din(self, dnn_config, deep_fea, name):
    cur_id, hist_id_col, seq_len = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len']

    seq_max_len = tf.shape(hist_id_col)[1]
    emb_dim = hist_id_col.shape[2]

    cur_ids = tf.tile(cur_id, [1, seq_max_len])
    cur_ids = tf.reshape(cur_ids,
                         tf.shape(hist_id_col))  # (B, seq_max_len, emb_dim)

    din_net = tf.concat(
        [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
        axis=-1)  # (B, seq_max_len, emb_dim*4)

    din_layer = dnn.DNN(
        dnn_config,
        self._l2_reg,
        name,
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    din_net = din_layer(din_net)
    scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

    seq_len = tf.expand_dims(seq_len, 1)
    mask = tf.sequence_mask(seq_len)
    padding = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(mask, scores, padding)  # [B, 1, seq_max_len]

    # Scale
    scores = tf.nn.softmax(scores)  # (B, 1, seq_max_len)
    hist_din_emb = tf.matmul(scores, hist_id_col)  # [B, 1, emb_dim]
    hist_din_emb = tf.reshape(hist_din_emb, [-1, emb_dim])  # [B, emb_dim]
    din_output = tf.concat([hist_din_emb, cur_id], axis=1)
    return din_output

  def build_predict_graph(self):
    tower_fea_arr = []
    for tower_id in range(self._tower_num):
      tower_fea = self._tower_features[tower_id]
      tower = self._model_config.towers[tower_id]
      tower_name = tower.input
      tower_fea = tf.layers.batch_normalization(
          tower_fea,
          training=self._is_training,
          trainable=True,
          name='%s_fea_bn' % tower_name)
      dnn_layer = dnn.DNN(tower.dnn, self._l2_reg, '%s_dnn' % tower_name,
                          self._is_training)
      tower_fea = dnn_layer(tower_fea)
      tower_fea_arr.append(tower_fea)

    for tower_id in range(self._din_tower_num):
      tower_fea = self._din_tower_features[tower_id]
      tower = self._model_config.din_towers[tower_id]
      tower_name = tower.input
      tower_fea = self.din(tower.dnn, tower_fea, name='%s_dnn' % tower_name)
      tower_fea_arr.append(tower_fea)

    all_fea = tf.concat(tower_fea_arr, axis=1)
    final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                              'final_dnn', self._is_training)
    all_fea = final_dnn_layer(all_fea)
    output = tf.layers.dense(all_fea, self._num_class, name='output')

    self._add_to_prediction_dict(output)

    return self._prediction_dict
