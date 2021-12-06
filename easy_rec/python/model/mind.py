# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers.capsule_layer import CapsuleLayer
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.model.match_model import MatchModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.mind_pb2 import MIND as MINDConfig
from easy_rec.python.protos.simi_pb2 import Similarity
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses
metrics = tf.metrics


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

    self._hist_seq_features = self._input_layer(
        self._feature_dict, 'hist', is_combine=False)
    self._user_features, _ = self._input_layer(self._feature_dict, 'user')
    self._item_features, _ = self._input_layer(self._feature_dict, 'item')

    # copy_obj so that any modification will not affect original config
    self.user_dnn = copy_obj(self._model_config.user_dnn)
    # copy_obj so that any modification will not affect original config
    self.item_dnn = copy_obj(self._model_config.item_dnn)

    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def build_predict_graph(self):
    capsule_layer = CapsuleLayer(self._model_config.capsule_config,
                                 self._is_training)

    time_id_fea = [
        x[0] for x in self._hist_seq_features if 'time_id/' in x[0].name
    ]
    time_id_fea = time_id_fea[0] if len(time_id_fea) > 0 else None

    hist_seq_feas = [
        x[0] for x in self._hist_seq_features if 'time_id/' not in x[0].name
    ]
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

    # batch_size x max_k x high_capsule_dim
    high_capsules, num_high_capsules = capsule_layer(hist_seq_feas,
                                                     hist_seq_len)
    # concatenate with user features
    user_features = tf.tile(
        tf.expand_dims(self._user_features, axis=1),
        [1, tf.shape(high_capsules)[1], 1])
    user_features = tf.concat([high_capsules, user_features], axis=2)
    num_user_dnn_layer = len(self.user_dnn.hidden_units)
    last_user_hidden = self.user_dnn.hidden_units.pop()
    user_dnn = dnn.DNN(self.user_dnn, self._l2_reg, 'user_dnn',
                       self._is_training)
    user_features = user_dnn(user_features)
    user_features = tf.layers.dense(
        inputs=user_features,
        units=last_user_hidden,
        kernel_regularizer=self._l2_reg,
        name='user_dnn/dnn_%d' % (num_user_dnn_layer - 1))

    num_item_dnn_layer = len(self.item_dnn.hidden_units)
    last_item_hidden = self.item_dnn.hidden_units.pop()
    item_dnn = dnn.DNN(self.item_dnn, self._l2_reg, 'item_dnn',
                       self._is_training)
    item_feature = item_dnn(self._item_features)
    item_feature = tf.layers.dense(
        inputs=item_feature,
        units=last_item_hidden,
        kernel_regularizer=self._l2_reg,
        name='item_dnn/dnn_%d' % (num_item_dnn_layer - 1))

    assert self._model_config.simi_func in [
        Similarity.COSINE, Similarity.INNER_PRODUCT
    ]

    if self._model_config.simi_func == Similarity.COSINE:
      item_feature = self.norm(item_feature)
      user_features = self.norm(user_features)

    # label guided attention
    # attention item features on high capsules vector
    batch_size = tf.shape(user_features)[0]
    pos_item_fea = item_feature[:batch_size]
    simi = tf.einsum('bhe,be->bh', user_features, pos_item_fea)
    simi = tf.pow(simi, self._model_config.simi_pow)
    simi_mask = tf.sequence_mask(num_high_capsules,
                                 self._model_config.capsule_config.max_k)

    user_features = user_features * tf.to_float(simi_mask[:, :, None])
    self._prediction_dict['user_features'] = user_features

    max_thresh = (tf.cast(simi_mask, tf.float32) * 2 - 1) * 1e32
    simi = tf.minimum(simi, max_thresh)
    simi = tf.nn.softmax(simi, axis=1)
    # important, but why ?
    simi = tf.stop_gradient(simi)
    user_tower_emb = tf.einsum('bhe,bh->be', user_features, simi)

    # calculate similarity between user_tower_emb and item_tower_emb
    item_tower_emb = item_feature
    user_item_sim = self.sim(user_tower_emb, item_tower_emb)
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

    if self._is_point_wise:
      y_pred = tf.reshape(y_pred, [-1])

    if self._loss_type in [LossType.CLASSIFICATION, LossType.SOFTMAX_CROSS_ENTROPY]:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

    self._prediction_dict['user_features'] = user_features
    self._prediction_dict['item_features'] = item_feature
    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.reduce_join(tf.as_string(user_features), axis=-1, separator=','),
        axis=-1,
        separator='|')
    self._prediction_dict['user_emb_num'] = num_high_capsules
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_tower_emb), axis=-1, separator=',')
    return self._prediction_dict

  def build_loss_graph(self):
    loss_dict = super(MIND, self).build_loss_graph()
    return loss_dict

  def _build_interest_metric(self):
    user_features = self._prediction_dict['user_features']
    user_features = self.norm(user_features)
    user_feature_num = self._prediction_dict['user_emb_num']

    user_feature_sum_sqr = tf.square(tf.reduce_sum(user_features, axis=1))
    user_feature_sqr_sum = tf.reduce_sum(tf.square(user_features), axis=1)
    simi = user_feature_sum_sqr - user_feature_sqr_sum

    # normalize by interest number
    simi = tf.reduce_sum(
        simi, axis=1) / tf.maximum(
            tf.to_float(user_feature_num * (user_feature_num - 1)), 1.0)
    
    # normalize by batch_size
    has_interest = tf.to_float(user_feature_num > 1)
    simi = (simi + 1) * has_interest / 2.0
    return metrics.mean(tf.reduce_sum(simi) / tf.maximum(tf.reduce_sum(has_interest), 1.0))

  def build_metric_graph(self, eval_config):
    # build interest metric
    metric_dict = { 'interest_similarity' : self._build_interest_metric() }
    if self._is_point_wise:
      metric_dict.update(self._build_point_wise_metric_graph(eval_config))
      return metric_dict

    recall_at_topk = []
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._loss_type == LossType.CLASSIFICATION
        if metric.topk not in recall_at_topk:
          recall_at_topk.append(metric.topk)

    # compute interest recall
    # [batch_size, num_interests, embed_dim]
    user_features = self._prediction_dict['user_features']
    # [?, embed_dim]
    item_feature = self._prediction_dict['item_features']
    batch_size = tf.shape(user_features)[0]
    hard_neg_indices = self._feature_dict.get('hard_neg_indices', None)

    if hard_neg_indices is not None:
      tf.logging.info('With hard negative examples')
      noclk_size = tf.shape(hard_neg_indices)[0]
      pos_item_emb, neg_item_emb, hard_neg_item_emb = tf.split(
          item_feature, [batch_size, -1, noclk_size], axis=0)
    else:
      pos_item_emb = item_feature[:batch_size]
      neg_item_emb = item_feature[batch_size:]
      hard_neg_item_emb = None

    # batch_size num_interest sample_neg_num
    pos_item_emb = tf.Print(pos_item_emb, [tf.shape(pos_item_emb),
        tf.shape(neg_item_emb), tf.shape(hard_neg_item_emb)], message='item_emb_shape')
    sample_item_sim = tf.einsum('bhe,ne->bhn', user_features, neg_item_emb)
    # batch_size sample_neg_num
    sample_item_sim = tf.reduce_max(sample_item_sim, axis=1)
    # batch_size num_interest
    pos_item_sim = tf.einsum('bhe,be->bh', user_features, pos_item_emb) 
    pos_item_sim = tf.reduce_sum(pos_item_sim, axis=1, keepdims=True) 
    
    sampled_logits = tf.concat([pos_item_sim, sample_item_sim], axis=1)
    sampled_lbls = tf.zeros_like(sampled_logits[:, :1], dtype=tf.int64)
    for topk in enumerate(recall_at_topk):
      metric_dict['interests_recall@%d' % topk] = \
            metrics.recall_at_k(labels=sampled_labels,
              predictions=sampled_logits, k=topk,
              name="interests_recall@%d" % topk)
    metric_dict['sampled_neg_acc'] = metrics.accuracy(sampled_lbls, sampled_logits)

    # batch_size num_interest
    if hard_neg_indices is not None:
      hard_neg_user_emb = tf.gather(user_features, hard_neg_indices[:, 0])
      hard_neg_sim = tf.einsum('nhe,ne->nh', hard_neg_user_emb, hard_neg_item_emb)
      hard_neg_sim = tf.reduce_max(hard_neg_sim, axis=1)
      max_num_neg = tf.reduce_max(hard_neg_indices[:, 1]) + 1 
      hard_neg_shape = tf.stack([tf.to_int64(batch_size), max_num_neg])
      hard_neg_sim = tf.scatter_nd(hard_neg_indices, tf.exp(hard_neg_sim), hard_neg_shape)
      hard_logits = tf.concat([pos_item_sim, hard_neg_sim], axis=1) 
      hard_lbls = tf.zeros_like(hard_logits[:, :1], dtype=tf.int64)
      metric_dict['hard_neg_acc'] = metrics.accuracy(hard_lbls, hard_logits)

    return metric_dict

  def get_outputs(self):
    if self._loss_type == LossType.CLASSIFICATION:
      return ['logits', 'user_emb', 'item_emb']
    elif self._loss_type == LossType.L2_LOSS:
      return ['y', 'user_emb', 'item_emb']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
