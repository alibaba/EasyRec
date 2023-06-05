# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.utils.shape_utils import get_shape_list
from easy_rec.python.layers.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import GRUCell
from easy_rec.python.utils.rnn_utils import VecAttGRUCell

# from tensorflow.python.keras.layers import Layer


class DIEN(object):

  def __init__(self, config, l2_reg, name='dien', **kwargs):
    # super(DIN, self).__init__(name=name, **kwargs)
    self.name = name
    self.l2_reg = l2_reg
    self.config = config

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

  def __call__(self, inputs, is_training=False, **kwargs):
    seq_features, target_feature = inputs
    seq_input = [seq_fea for seq_fea, _ in seq_features]
    hist_id_col = tf.concat(seq_input, axis=-1)
    batch_size, max_seq_len, _ = get_shape_list(hist_id_col, 3)

    query = target_feature[:batch_size, ...]
    target_emb_size = target_feature.shape.as_list()[-1]
    seq_emb_size = hist_id_col.shape.as_list()[-1]
    if target_emb_size != seq_emb_size:
      logging.info(
          '<din> the embedding size of sequence [%d] and target item [%d] is not equal'
          ' in feature group: %s', seq_emb_size, target_emb_size, self.name)
      if target_emb_size < seq_emb_size:
        query = tf.pad(target_feature,
                       [[0, 0], [0, seq_emb_size - target_emb_size]])
      else:
        assert False, 'the embedding size of target item is larger than the one of sequence'

    seq_len = seq_features[0][1]
    rnn_outputs,_ = dynamic_rnn(GRUCell(seq_emb_size),
       inputs=hist_id_col, sequence_length=seq_len,
       dtype=tf.float32, scope='%s/dien/gru1' % self.name)

    seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.bool)
    if self.config.aux_loss_weight > 0 and is_training:
      logging.info('add dien aux_loss[weight=%.3f]' % self.config.aux_loss_weight)
      neg_cur_id = target_feature[batch_size:, ...]
      neg_cur_id = tf.tile(
          neg_cur_id[None, :, :], multiples=[batch_size, 1, 1])
      aux_loss = self.auxiliary_loss(
          rnn_outputs[:, :-1, :],
          hist_id_col[:, 1:, :target_emb_size],
          neg_cur_id[:, 1:, :],
          seq_mask[:, 1:],
          stag=self.name + '/dien/aux_loss') * self.config.aux_loss_weight
    else:
      aux_loss = None

    queries = tf.tile(tf.expand_dims(query, 1), [1, max_seq_len, 1])
    din_all = tf.concat([queries, rnn_outputs, queries - rnn_outputs, queries * rnn_outputs],
                        axis=-1)
    din_layer = dnn.DNN(
        self.config.attention_dnn,
        self.l2_reg,
        self.name + '/din_attention',
        training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    output = din_layer(din_all)  # [B, L, 1]
    scores = tf.transpose(output, [0, 2, 1])  # [B, 1, L]

    seq_mask = tf.expand_dims(seq_mask, 1)
    paddings = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(seq_mask, scores, paddings)  # [B, 1, L]
    if self.config.attention_normalizer == 'softmax':
      scores = tf.nn.softmax(scores)  # (B, 1, L)
    elif self.config.attention_normalizer == 'sigmoid':
      scores = scores / (seq_emb_size**0.5)
      scores = tf.nn.sigmoid(scores)
    else:
      raise ValueError('unsupported attention normalizer: ' +
                       self.config.attention_normalizer)

    _, final_state2 = dynamic_rnn(
        VecAttGRUCell(seq_emb_size),
        inputs=rnn_outputs,
        att_scores=tf.transpose(scores, [0, 2, 1]),
        sequence_length=seq_len,
        dtype=tf.float32,
        scope='%s/dien/gru2' % self.name)

    # if target_emb_size < seq_emb_size:
    #   hist_id_col = hist_id_col[:, :, :target_emb_size]  # [B, L, E]

    output = tf.squeeze(tf.matmul(scores, rnn_outputs), axis=[1])
    item_his_eb_sum = tf.reduce_sum(hist_id_col, 1)
    output = tf.concat([output, target_feature, item_his_eb_sum,
          target_feature * item_his_eb_sum[:, :target_emb_size], final_state2], axis=-1)
    logging.info('dien output shape: ' + str(output.shape))
    return output, aux_loss
