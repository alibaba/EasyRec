# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell import GRUCell

from easy_rec.python.layers import dnn
from easy_rec.python.layers.rnn import dynamic_rnn
from easy_rec.python.utils import shape_utils
from easy_rec.python.utils.rnn_utils import VecAttGRUCell

# from easy_rec.python.utils.shape_utils import get_shape_list
# from tensorflow.python.keras.layers import Layer


class DIEN(object):

  def __init__(self, config, l2_reg, name='dien', features=None, **kwargs):
    # super(DIN, self).__init__(name=name, **kwargs)
    self.name = name
    self.l2_reg = l2_reg
    self.config = config
    self.features = features

  def auxiliary_loss(self,
                     h_states,
                     click_seq,
                     noclick_seq,
                     mask,
                     pos_item_ids,
                     neg_item_ids,
                     stag=None):

    mask = tf.cast(mask, tf.float32)
    click_input = tf.concat([h_states, click_seq], -1)
    # batch_size, neg_num, max_seq_len, embed_dim
    h_states_neg = tf.tile(h_states[:, None, :, :],
                           [1, self.config.negative_num, 1, 1])
    # batch_size, neg_num, max_seq_len
    mask_neg = tf.tile(mask[:, None, :], [1, self.config.negative_num, 1])
    # batch_size, neg_num, max_seq_len, embed_dim * 2
    noclick_input = tf.concat([h_states_neg, noclick_seq], -1)
    # batch_size, neg_num, max_seq_len
    pos_item_ids = tf.tile(pos_item_ids[:, None, :],
                           [1, self.config.negative_num, 1])

    mask_neg_eq_pos = (1 - tf.to_float(tf.equal(pos_item_ids, neg_item_ids)))
    tmp_div = math_ops.reduce_mean(mask_neg)
    mask_neg = mask_neg_eq_pos * mask_neg
    tf.summary.scalar('dien/aux_loss/neg_not_eq_pos',
                      math_ops.reduce_mean(mask_neg_eq_pos))
    tf.summary.scalar('dien/aux_loss/mask_neg', math_ops.reduce_mean(mask_neg))
    tf.summary.scalar('dien/aux_loss/neg_not_eq_pos_normed',
                      math_ops.reduce_mean(mask_neg) / (tmp_div + 1e-10))
    click_prop = self.auxiliary_net(click_input, stag=stag)[:, :, 0]
    noclick_prop = self.auxiliary_net(noclick_input, stag=stag)[:, :, :, 0]
    click_loss = -tf.log(click_prop) * mask
    noclick_loss = -tf.log(1.0 - noclick_prop) * mask_neg
    loss = tf.reduce_mean(click_loss) + tf.reduce_mean(noclick_loss)
    loss = tf.Print(
        loss, [
            tf.reduce_mean(click_loss),
            tf.reduce_mean(noclick_loss),
            tf.shape(h_states),
            tf.shape(click_seq),
            tf.shape(noclick_seq),
            tf.shape(mask),
            tf.shape(mask_neg),
            tf.shape(pos_item_ids),
            tf.shape(neg_item_ids), tmp_div
        ],
        message='aux_loss')
    return loss

  def auxiliary_net(self, in_fea, stag='auxiliary_net'):
    bn1 = tf.layers.batch_normalization(
        inputs=in_fea, name=stag + '/bn1', reuse=tf.AUTO_REUSE)
    dnn1 = tf.layers.dense(
        bn1, 100, activation=None, name=stag + '/f1', reuse=tf.AUTO_REUSE)
    dnn1 = tf.nn.sigmoid(dnn1)
    dnn2 = tf.layers.dense(
        dnn1, 50, activation=None, name=stag + '/f2', reuse=tf.AUTO_REUSE)
    dnn2 = tf.nn.sigmoid(dnn2)
    dnn3 = tf.layers.dense(
        dnn2, 2, activation=None, name=stag + '/f3', reuse=tf.AUTO_REUSE)
    y_hat = tf.nn.softmax(dnn3) + 0.00000001
    return y_hat

  def __call__(self, inputs, is_training=False, **kwargs):
    seq_features, target_feature = inputs
    seq_input = [seq_fea for seq_fea, _ in seq_features]
    hist_id_col = tf.concat(seq_input, axis=-1)
    seq_len = seq_features[0][1]
    hist_id_col = array_ops.reverse_sequence(
        hist_id_col,
        seq_len,
        seq_axis=1,
        batch_axis=0,
        name='%s/seq_reverse' % self.name)

    # batch_size, max_seq_len, _ = get_shape_list(hist_id_col, 3)
    batch_size = tf.shape(hist_id_col)[0]
    max_seq_len = tf.shape(hist_id_col)[1]

    tf.summary.scalar('max_seq_len', max_seq_len)
    tf.summary.scalar('avg_seq_len', math_ops.reduce_mean(tf.to_float(seq_len)))

    # target_feature = tf.Print(target_feature, [tf.shape(target_feature),
    #     batch_size, max_seq_len], message='target_feature_shape')
    query = target_feature[:batch_size, ...]
    target_emb_size = target_feature.shape.as_list()[-1]
    seq_emb_size = hist_id_col.shape.as_list()[-1]
    if target_emb_size != seq_emb_size:
      logging.info(
          '<din> the embedding size of sequence [%d] and target item [%d] is not equal'
          ' in feature group: %s', seq_emb_size, target_emb_size, self.name)
      if target_emb_size < seq_emb_size:
        query = tf.pad(query, [[0, 0], [0, seq_emb_size - target_emb_size]])
      else:
        assert False, 'the embedding size of target item is larger than the one of sequence'

    rnn_outputs, _ = dynamic_rnn(
        GRUCell(seq_emb_size),
        inputs=hist_id_col,
        sequence_length=seq_len,
        dtype=tf.float32,
        scope='%s/dien/gru1' % self.name)

    seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.bool)
    if self.config.aux_loss_weight > 0 and is_training:
      logging.info('add dien aux_loss[weight=%.3f]' %
                   self.config.aux_loss_weight)
      neg_item_embed = target_feature[batch_size:, ...]
      neg_item_num = self.config.negative_num * (max_seq_len - 1)
      neg_item_embed = shape_utils.pad_or_clip_tensor(neg_item_embed,
                                                      neg_item_num)
      neg_item_embed = tf.tile(
          neg_item_embed[None, :, :], multiples=[batch_size, 1, 1])
      neg_item_embed = tf.reshape(neg_item_embed, [
          batch_size, self.config.negative_num, max_seq_len - 1, target_emb_size
      ])

      item_ids = self.features[self.config.item_id]
      neg_item_ids = item_ids[batch_size:]
      neg_item_ids = shape_utils.pad_or_clip_tensor(neg_item_ids, neg_item_num)
      # neg_item_ids = tf.Print(
      #     neg_item_ids, [tf.shape(item_ids), batch_size, max_seq_len],
      #     message='dump_item_ids_shape')
      # neg_item_ids = neg_item_ids[:(self.config.negative_num *
      #                               (max_seq_len - 1))]
      neg_item_ids = tf.tile(neg_item_ids[None, :], [batch_size, 1])
      neg_item_ids = tf.reshape(
          neg_item_ids, [batch_size, self.config.negative_num, max_seq_len - 1])
      pos_item_ids = tf.sparse.to_dense(
          self.features[self.config.seq_item_id], default_value='')
      aux_loss = self.auxiliary_loss(
          rnn_outputs[:, :-1, :],
          hist_id_col[:, 1:, :target_emb_size],
          neg_item_embed,
          seq_mask[:, 1:],
          pos_item_ids[:, 1:],
          neg_item_ids,
          stag=self.name + '/dien/aux_loss') * self.config.aux_loss_weight
    else:
      aux_loss = None

    queries = tf.tile(tf.expand_dims(query, 1), [1, max_seq_len, 1])
    din_all = tf.concat(
        [queries, rnn_outputs, queries - rnn_outputs, queries * rnn_outputs],
        axis=-1)
    din_layer = dnn.DNN(
        self.config.attention_dnn,
        self.l2_reg,
        self.name + '/din_attention',
        is_training,
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
    output = tf.concat([
        output, target_feature[:batch_size, ...], item_his_eb_sum,
        target_feature[:batch_size, ...] * item_his_eb_sum[:, :target_emb_size],
        final_state2
    ],
                       axis=-1)  # noqa: E126
    logging.info('dien output shape: ' + str(output.shape))
    return output, aux_loss
