# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers.capsule_layer import CapsuleLayer
from easy_rec.python.model.match_model import MatchModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.mind_pb2 import MIND as MINDConfig
from easy_rec.python.protos.simi_pb2 import Similarity
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses


class MIND(MatchModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MIND, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert self._model_config.WhichOneof('model') == 'mind', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.mind

    self._hist_seq_features, _, _ = self._input_layer(
        self._feature_dict, 'hist', is_combine=False)
    self._user_features, _ = self._input_layer(self._feature_dict, 'user')
    self._item_features, _ = self._input_layer(self._feature_dict, 'item')

    # copy_obj so that any modification will not affect original config
    self.user_dnn = copy_obj(self._model_config.user_dnn)
    # copy_obj so that any modification will not affect original config
    self.item_dnn = copy_obj(self._model_config.item_dnn)
    # copy obj so that any modification will not affect original config
    self.concat_dnn = copy_obj(self._model_config.concat_dnn)

    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def build_predict_graph(self):
    capsule_layer = CapsuleLayer(self._model_config.capsule_config,
                                 self._is_training)

    if self._model_config.time_id_fea:
      time_id_fea = [
          x[0]
          for x in self._hist_seq_features
          if self._model_config.time_id_fea in x[0].name
      ]
      logging.info('time_id_fea is set(%s), find num: %d' %
                   (self._model_config.time_id_fea, len(time_id_fea)))
    else:
      time_id_fea = []
    time_id_fea = time_id_fea[0] if len(time_id_fea) > 0 else None

    if time_id_fea is not None:
      hist_seq_feas = [
          x[0]
          for x in self._hist_seq_features
          if self._model_config.time_id_fea not in x[0].name
      ]
    else:
      hist_seq_feas = [x[0] for x in self._hist_seq_features]

    # it is assumed that all hist have the same length
    hist_seq_len = self._hist_seq_features[0][1]

    if self._model_config.user_seq_combine == MINDConfig.SUM:
      # sum pooling over the features
      hist_embed_dims = [x.get_shape()[-1] for x in hist_seq_feas]
      for i in range(1, len(hist_embed_dims)):
        assert hist_embed_dims[i] == hist_embed_dims[0], \
            'all hist seq must have the same embedding shape, but: %s' \
            % str(hist_embed_dims)
      hist_seq_feas = tf.add_n(hist_seq_feas) / len(hist_seq_feas)
    else:
      hist_seq_feas = tf.concat(hist_seq_feas, axis=2)

    if self._model_config.HasField('pre_capsule_dnn') and \
        len(self._model_config.pre_capsule_dnn.hidden_units) > 0:
      pre_dnn_layer = dnn.DNN(self._model_config.pre_capsule_dnn, self._l2_reg,
                              'pre_capsule_dnn', self._is_training)
      hist_seq_feas = pre_dnn_layer(hist_seq_feas)

    if time_id_fea is not None:
      assert time_id_fea.get_shape(
      )[-1] == 1, 'time_id must have only embedding_size of 1'
      time_id_mask = tf.sequence_mask(hist_seq_len, tf.shape(time_id_fea)[1])
      time_id_mask = (tf.cast(time_id_mask, tf.float32) * 2 - 1) * 1e32
      time_id_fea = tf.minimum(time_id_fea, time_id_mask[:, :, None])
      hist_seq_feas = hist_seq_feas * tf.nn.softmax(time_id_fea, axis=1)

    tf.summary.histogram('hist_seq_len', hist_seq_len)

    # batch_size x max_k x high_capsule_dim
    high_capsules, num_high_capsules = capsule_layer(hist_seq_feas,
                                                     hist_seq_len)

    tf.summary.histogram('num_high_capsules', num_high_capsules)

    # high_capsules = tf.layers.batch_normalization(
    #     high_capsules, training=self._is_training,
    #     trainable=True, name='capsule_bn')
    # high_capsules = high_capsules * 0.1

    tf.summary.scalar('high_capsules_norm',
                      tf.reduce_mean(tf.norm(high_capsules, axis=-1)))
    tf.summary.scalar('num_high_capsules',
                      tf.reduce_mean(tf.to_float(num_high_capsules)))

    user_features = tf.layers.batch_normalization(
        self._user_features,
        training=self._is_training,
        trainable=True,
        name='user_fea_bn')
    user_dnn = dnn.DNN(self.user_dnn, self._l2_reg, 'user_dnn',
                       self._is_training)
    user_features = user_dnn(user_features)

    tf.summary.scalar('user_features_norm',
                      tf.reduce_mean(tf.norm(self._user_features, axis=-1)))

    # concatenate with user features
    user_features_tile = tf.tile(user_features[:, None, :],
                                 [1, tf.shape(high_capsules)[1], 1])
    user_interests = tf.concat([high_capsules, user_features_tile], axis=2)

    num_concat_dnn_layer = len(self.concat_dnn.hidden_units)
    last_hidden = self.concat_dnn.hidden_units.pop()
    concat_dnn = dnn.DNN(self.concat_dnn, self._l2_reg, 'concat_dnn',
                         self._is_training)
    user_interests = concat_dnn(user_interests)
    user_interests = tf.layers.dense(
        inputs=user_interests,
        units=last_hidden,
        kernel_regularizer=self._l2_reg,
        name='concat_dnn/dnn_%d' % (num_concat_dnn_layer - 1))

    num_item_dnn_layer = len(self.item_dnn.hidden_units)
    last_item_hidden = self.item_dnn.hidden_units.pop()
    item_dnn = dnn.DNN(self.item_dnn, self._l2_reg, 'item_dnn',
                       self._is_training)
    item_tower_emb = item_dnn(self._item_features)
    item_tower_emb = tf.layers.dense(
        inputs=item_tower_emb,
        units=last_item_hidden,
        kernel_regularizer=self._l2_reg,
        name='item_dnn/dnn_%d' % (num_item_dnn_layer - 1))

    assert self._model_config.simi_func in [
        Similarity.COSINE, Similarity.INNER_PRODUCT
    ]

    if self._model_config.simi_func == Similarity.COSINE:
      item_tower_emb = self.norm(item_tower_emb)
      user_interests = self.norm(user_interests)

    # label guided attention
    # attention item features on high capsules vector
    batch_size = tf.shape(user_interests)[0]
    pos_item_fea = item_tower_emb[:batch_size]
    simi = tf.einsum('bhe,be->bh', user_interests, pos_item_fea)
    tf.summary.histogram('interest_item_simi/pre_scale',
                         tf.reduce_max(simi, axis=1))
    # simi = tf.Print(simi, [tf.reduce_max(simi, axis=1), tf.reduce_min(simi, axis=1)], message='simi_max_0')
    # simi = tf.pow(simi, self._model_config.simi_pow)
    simi = simi * self._model_config.simi_pow
    tf.summary.histogram('interest_item_simi/scaled',
                         tf.reduce_max(simi, axis=1))
    # simi = tf.Print(simi, [tf.reduce_max(simi, axis=1), tf.reduce_min(simi, axis=1)], message='simi_max')
    simi_mask = tf.sequence_mask(num_high_capsules,
                                 self._model_config.capsule_config.max_k)

    user_interests = user_interests * tf.to_float(simi_mask[:, :, None])
    self._prediction_dict['user_interests'] = user_interests

    max_thresh = (tf.cast(simi_mask, tf.float32) * 2 - 1) * 1e32
    simi = tf.minimum(simi, max_thresh)
    simi = tf.nn.softmax(simi, axis=1)
    tf.summary.histogram('interest_item_simi/softmax',
                         tf.reduce_max(simi, axis=1))

    if self._model_config.simi_pow >= 100:
      logging.info(
          'simi_pow=%d, will change to argmax, only use the most similar interests for calculate loss.'
          % self._model_config.simi_pow)
      simi_max_id = tf.argmax(simi, axis=1)
      simi = tf.one_hot(simi_max_id, tf.shape(simi)[1], dtype=tf.float32)

    user_tower_emb = tf.einsum('bhe,bh->be', user_interests, simi)

    # calculate similarity between user_tower_emb and item_tower_emb
    user_item_sim = self.sim(user_tower_emb, item_tower_emb)
    if self._model_config.scale_simi:
      sim_w = tf.get_variable(
          'sim_w',
          dtype=tf.float32,
          shape=(1),
          initializer=tf.ones_initializer())
      sim_b = tf.get_variable(
          'sim_b',
          dtype=tf.float32,
          shape=(1),
          initializer=tf.zeros_initializer())
      y_pred = user_item_sim * tf.abs(sim_w) + sim_b
    else:
      y_pred = user_item_sim

    if self._is_point_wise:
      y_pred = tf.reshape(y_pred, [-1])

    if self._loss_type == LossType.CLASSIFICATION:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      y_pred = self._mask_in_batch(y_pred)
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.softmax(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

    self._prediction_dict['high_capsules'] = high_capsules
    self._prediction_dict['user_interests'] = user_interests
    self._prediction_dict['user_tower_emb'] = user_tower_emb
    self._prediction_dict['item_tower_emb'] = item_tower_emb
    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.reduce_join(tf.as_string(user_interests), axis=-1, separator=','),
        axis=-1,
        separator='|')
    self._prediction_dict['user_emb_num'] = num_high_capsules
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_tower_emb), axis=-1, separator=',')

    if self._labels is not None:
      # for summary purpose
      batch_simi, batch_capsule_simi = self._build_interest_simi()
      # self._prediction_dict['probs'] = tf.Print(self._prediction_dict['probs'],
      #     [batch_simi, batch_capsule_simi], message='batch_simi')
      self._prediction_dict['interests_simi'] = batch_simi
    return self._prediction_dict

  def build_loss_graph(self):
    loss_dict = super(MIND, self).build_loss_graph()
    if self._model_config.max_interests_simi < 1.0:
      loss_dict['reg_interest_simi'] = tf.nn.relu(
          self._prediction_dict['interests_simi'] -
          self._model_config.max_interests_simi)
    return loss_dict

  def _build_interest_simi(self):
    user_emb_num = self._prediction_dict['user_emb_num']
    high_capsule_mask = tf.sequence_mask(
        user_emb_num, self._model_config.capsule_config.max_k)

    user_interests = self._prediction_dict['user_interests']
    high_capsule_mask = tf.to_float(high_capsule_mask[:, :, None])
    user_interests = self.norm(user_interests) * high_capsule_mask

    user_feature_sum_sqr = tf.square(tf.reduce_sum(user_interests, axis=1))
    user_feature_sqr_sum = tf.reduce_sum(tf.square(user_interests), axis=1)
    interest_simi = user_feature_sum_sqr - user_feature_sqr_sum

    high_capsules = self._prediction_dict['high_capsules']
    high_capsules = self.norm(high_capsules) * high_capsule_mask
    high_capsule_sum_sqr = tf.square(tf.reduce_sum(high_capsules, axis=1))
    high_capsule_sqr_sum = tf.reduce_sum(tf.square(high_capsules), axis=1)
    high_capsule_simi = high_capsule_sum_sqr - high_capsule_sqr_sum

    # normalize by interest number
    interest_div = tf.maximum(
        tf.to_float(user_emb_num * (user_emb_num - 1)), 1.0)
    interest_simi = tf.reduce_sum(interest_simi, axis=1) / interest_div

    high_capsule_simi = tf.reduce_sum(high_capsule_simi, axis=1) / interest_div

    # normalize by batch_size
    multi_interest = tf.to_float(user_emb_num > 1)
    sum_interest_simi = tf.reduce_sum(
        (interest_simi + 1) * multi_interest) / 2.0
    sum_div = tf.maximum(tf.reduce_sum(multi_interest), 1.0)
    avg_interest_simi = sum_interest_simi / sum_div

    sum_capsule_simi = tf.reduce_sum(
        (high_capsule_simi + 1) * multi_interest) / 2.0
    avg_capsule_simi = sum_capsule_simi / sum_div

    tf.summary.scalar('interest_similarity', avg_interest_simi)
    tf.summary.scalar('capsule_similarity', avg_capsule_simi)
    return avg_interest_simi, avg_capsule_simi

  def build_metric_graph(self, eval_config):
    from easy_rec.python.core.easyrec_metrics import metrics_tf as metrics
    # build interest metric
    interest_simi, capsule_simi = self._build_interest_simi()
    metric_dict = {
        'interest_similarity': metrics.mean(interest_simi),
        'capsule_similarity': metrics.mean(capsule_simi)
    }
    if self._is_point_wise:
      metric_dict.update(self._build_point_wise_metric_graph(eval_config))
      return metric_dict

    recall_at_topks = []
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._loss_type in [
            LossType.CLASSIFICATION, LossType.SOFTMAX_CROSS_ENTROPY
        ]
        if metric.recall_at_topk.topk not in recall_at_topks:
          recall_at_topks.append(metric.recall_at_topk.topk)

    # compute interest recall
    # [batch_size, num_interests, embed_dim]
    user_interests = self._prediction_dict['user_interests']
    # [?, embed_dim]
    item_tower_emb = self._prediction_dict['item_tower_emb']
    batch_size = tf.shape(user_interests)[0]
    # [?, 2] first dimension is the sample_id in batch
    # second dimension is the neg_id with respect to the sample
    hard_neg_indices = self._feature_dict.get('hard_neg_indices', None)

    if hard_neg_indices is not None:
      logging.info('With hard negative examples')
      noclk_size = tf.shape(hard_neg_indices)[0]
      simple_item_emb, hard_neg_item_emb = tf.split(
          item_tower_emb, [-1, noclk_size], axis=0)
    else:
      simple_item_emb = item_tower_emb
      hard_neg_item_emb = None

    # batch_size num_interest sample_neg_num
    simple_item_sim = tf.einsum('bhe,ne->bhn', user_interests, simple_item_emb)
    # batch_size sample_neg_num
    simple_item_sim = tf.reduce_max(simple_item_sim, axis=1)
    simple_lbls = tf.cast(tf.range(tf.shape(user_interests)[0]), tf.int64)

    # labels = tf.zeros_like(logits[:, :1], dtype=tf.int64)
    pos_indices = tf.range(batch_size)
    pos_indices = tf.concat([pos_indices[:, None], pos_indices[:, None]],
                            axis=1)
    pos_item_sim = tf.gather_nd(simple_item_sim[:batch_size, :batch_size],
                                pos_indices)

    simple_item_sim_v2 = tf.concat(
        [pos_item_sim[:, None], simple_item_sim[:, batch_size:]], axis=1)
    simple_lbls_v2 = tf.zeros_like(simple_item_sim_v2[:, :1], dtype=tf.int64)

    for topk in recall_at_topks:
      metric_dict['interests_recall@%d' % topk] = metrics.recall_at_k(
          labels=simple_lbls,
          predictions=simple_item_sim,
          k=topk,
          name='interests_recall_at_%d' % topk)
      metric_dict['interests_neg_sam_recall@%d' % topk] = metrics.recall_at_k(
          labels=simple_lbls_v2,
          predictions=simple_item_sim_v2,
          k=topk,
          name='interests_recall_neg_sam_at_%d' % topk)

    logits = self._prediction_dict['logits']
    pos_item_logits = tf.gather_nd(logits[:batch_size, :batch_size],
                                   pos_indices)
    logits_v2 = tf.concat([pos_item_logits[:, None], logits[:, batch_size:]],
                          axis=1)
    labels_v2 = tf.zeros_like(logits_v2[:, :1], dtype=tf.int64)

    for topk in recall_at_topks:
      metric_dict['recall@%d' % topk] = metrics.recall_at_k(
          labels=simple_lbls,
          predictions=logits,
          k=topk,
          name='recall_at_%d' % topk)
      metric_dict['recall_neg_sam@%d' % topk] = metrics.recall_at_k(
          labels=labels_v2,
          predictions=logits_v2,
          k=topk,
          name='recall_neg_sam_at_%d' % topk)
      eval_logits = logits[:, :batch_size]
      eval_logits = tf.cond(
          batch_size < topk, lambda: tf.pad(
              eval_logits, [[0, 0], [0, topk - batch_size]],
              mode='CONSTANT',
              constant_values=-1e32,
              name='pad_eval_logits'), lambda: eval_logits)
      metric_dict['recall_in_batch@%d' % topk] = metrics.recall_at_k(
          labels=simple_lbls,
          predictions=eval_logits,
          k=topk,
          name='recall_in_batch_at_%d' % topk)

    # batch_size num_interest
    if hard_neg_indices is not None:
      hard_neg_user_emb = tf.gather(user_interests, hard_neg_indices[:, 0])
      hard_neg_sim = tf.einsum('nhe,ne->nh', hard_neg_user_emb,
                               hard_neg_item_emb)
      hard_neg_sim = tf.reduce_max(hard_neg_sim, axis=1)
      max_num_neg = tf.reduce_max(hard_neg_indices[:, 1]) + 1
      hard_neg_shape = tf.stack([tf.to_int64(batch_size), max_num_neg])
      hard_neg_mask = tf.scatter_nd(
          hard_neg_indices,
          tf.ones_like(hard_neg_sim, dtype=tf.float32),
          shape=hard_neg_shape)
      hard_neg_sim = tf.scatter_nd(hard_neg_indices, hard_neg_sim,
                                   hard_neg_shape)
      hard_neg_sim = hard_neg_sim - (1 - hard_neg_mask) * 1e32

      hard_logits = tf.concat([pos_item_logits[:, None], hard_neg_sim], axis=1)
      hard_lbls = tf.zeros_like(hard_logits[:, :1], dtype=tf.int64)
      metric_dict['hard_neg_acc'] = metrics.accuracy(
          hard_lbls, tf.argmax(hard_logits, axis=1))

    return metric_dict

  def get_outputs(self):
    if self._loss_type == LossType.CLASSIFICATION:
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_emb_num',
          'user_interests', 'item_tower_emb'
      ]
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      self._prediction_dict['logits'] = tf.squeeze(
          self._prediction_dict['logits'], axis=-1)
      self._prediction_dict['probs'] = tf.nn.sigmoid(
          self._prediction_dict['logits'])
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_emb_num',
          'user_interests', 'item_tower_emb'
      ]
    elif self._loss_type == LossType.L2_LOSS:
      return [
          'y', 'user_emb', 'item_emb', 'user_emb_num', 'user_interests',
          'item_tower_emb'
      ]
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
