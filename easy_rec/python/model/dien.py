# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers import seq_input_layer
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.model.rnn import dynamic_rnn
from easy_rec.python.utils.rnn_utils import VecAttGRUCell

from easy_rec.python.protos.dien_pb2 import DIEN as DIENConfig  # NOQA

if tf.__version__ >= '2.0':
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
  tf = tf.compat.v1
else:
  losses = tf.losses
  metrics = tf.metrics


class DIEN(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DIEN, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    self._seq_input_layer = seq_input_layer.SeqInputLayer(
        feature_configs,
        model_config.seq_att_groups,
        embedding_regularizer=self._emb_reg,
        ev_params=self._global_ev_params)
    assert self._model_config.WhichOneof('model') == 'dien', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dien
    assert isinstance(self._model_config, DIENConfig)

    self.user_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.item_feature, _ = self._input_layer(self._feature_dict, 'item')

    self._din_tower_features = []
    self._din_tower_num = len(self._model_config.din_towers)
    for tower_id in range(self._din_tower_num):
      tower = self._model_config.din_towers[tower_id]
      tower_feature = self._seq_input_layer(self._feature_dict, tower.input)
      regularizers.apply_regularization(
          self._emb_reg, weights_list=[tower_feature['hist_seq_emb']])
      self._din_tower_features.append(tower_feature)

    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
    mask = tf.cast(mask, tf.float32)
    click_input_ = tf.concat([h_states, click_seq], -1)
    noclick_input_ = tf.concat([h_states, noclick_seq], -1)
    click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
    noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
    click_loss_ = -tf.reshape(
        tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
    noclick_loss_ = -tf.reshape(
        tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
    loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
    return loss_

  def auxiliary_net(self, in_, stag='auxiliary_net'):
    bn1 = tf.layers.batch_normalization(
        inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
    dnn1 = tf.layers.dense(
        bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
    dnn1 = tf.nn.sigmoid(dnn1)
    dnn2 = tf.layers.dense(
        dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
    dnn2 = tf.nn.sigmoid(dnn2)
    dnn3 = tf.layers.dense(
        dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
    y_hat = tf.nn.softmax(dnn3) + 0.00000001
    return y_hat

  def dien(self, dnn_config, deep_fea, name):
    cur_id, hist_id_col, seq_len = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len']
    seq_max_len = tf.shape(hist_id_col)[1]
    emb_dim = hist_id_col.shape[2]
    batch_size = tf.shape(hist_id_col)[0]
    mask = tf.sequence_mask(seq_len)

    neg_cur_id = cur_id[batch_size:, ...]  # for negative sampler # [50 32]
    cur_id = cur_id[:batch_size, ...]

    rnn_outputs, _ = dynamic_rnn(
        GRUCell(emb_dim),
        inputs=hist_id_col,
        sequence_length=seq_len,
        dtype=tf.float32,
        scope='gru1')

    neg_cur_id = tf.tile(
        neg_cur_id[tf.newaxis, :, :], multiples=[batch_size, 1, 1])
    aux_loss_1 = self.auxiliary_loss(
        rnn_outputs[:, :-1, :],
        hist_id_col[:, 1:, :],
        neg_cur_id[:, 1:, :],
        mask[:, 1:],
        stag='gru')
    self.aux_loss = aux_loss_1

    hist_id_col_ = rnn_outputs

    cur_ids = tf.tile(cur_id, [1, seq_max_len])
    cur_ids = tf.reshape(cur_ids, tf.shape(hist_id_col))

    din_net = tf.concat(
        [cur_ids, hist_id_col_, cur_ids - hist_id_col_, cur_ids * hist_id_col_],
        axis=-1)

    din_layer = dnn.DNN(
        dnn_config,
        self._l2_reg,
        name,
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    din_net = din_layer(din_net)
    scores = tf.reshape(din_net, [-1, seq_max_len])

    padding = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(mask, scores, padding)

    # Scale
    scores = tf.nn.softmax(scores)
    hist_din_emb = tf.reduce_sum(
        scores[:, :, tf.newaxis] * hist_id_col_, axis=1)
    hist_din_emb = tf.reshape(hist_din_emb, [-1, emb_dim])
    rnn_outputs2, final_state2 = dynamic_rnn(
        VecAttGRUCell(emb_dim),
        inputs=rnn_outputs,
        att_scores=tf.expand_dims(scores, -1),
        sequence_length=seq_len,
        dtype=tf.float32,
        scope='gru2')
    item_his_eb_sum = tf.reduce_sum(hist_id_col, 1)
    output = tf.concat([
        hist_din_emb, cur_id, item_his_eb_sum, cur_id * item_his_eb_sum,
        final_state2
    ], 1)
    return output

  def build_predict_graph(self):
    din_fea_arr = []
    for tower_id in range(self._din_tower_num):
      tower_fea = self._din_tower_features[tower_id]
      tower = self._model_config.din_towers[tower_id]
      tower_name = tower.input
      tower_fea = self.dien(tower.dnn, tower_fea, name='%s_dnn' % tower_name)
      din_fea_arr.append(tower_fea)
    din_fea_arr = tf.concat(din_fea_arr, axis=-1)
    batch_size = tf.shape(self.user_feature)[0]
    all_fea = tf.concat(
        [self.user_feature, self.item_feature[:batch_size], din_fea_arr], 1)

    dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
                        self._is_training)
    all_fea = dnn_layer(all_fea)
    output = tf.layers.dense(all_fea, 1, name='output')
    self._add_to_prediction_dict(output)
    return self._prediction_dict

  def build_loss_graph(self):
    self._loss_dict[
        'aux_loss'] = self._model_config.aux_loss_weight * self.aux_loss

    self._loss_dict.update(
        self._build_loss_impl(
            self._loss_type,
            label_name=self._label_name,
            loss_weight=self._sample_weight,
            num_class=self._num_class))

    return self._loss_dict

  def get_outputs(self):
    outputs = super(DIEN, self).get_outputs()
    return outputs
