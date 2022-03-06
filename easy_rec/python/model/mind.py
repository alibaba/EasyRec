# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers.capsule_layer import CapsuleLayer
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.mind_pb2 import MIND as MINDConfig
from easy_rec.python.protos.simi_pb2 import Similarity
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses
metrics = tf.metrics


class MIND(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MIND, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class
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

    if self._labels is not None:
      self._labels = list(self._labels.values())
      if self._loss_type == LossType.CLASSIFICATION:
        self._labels[0] = tf.cast(self._labels[0], tf.int64)
      elif self._loss_type == LossType.L2_LOSS:
        self._labels[0] = tf.cast(self._labels[0], tf.float32)

    if self._loss_type == LossType.CLASSIFICATION:
      assert self._num_class == 1

  def sim(self, user_emb, item_emb):
    user_item_sim = tf.reduce_sum(
        tf.multiply(user_emb, item_emb), axis=1, keep_dims=True)
    return user_item_sim

  def norm(self, fea):
    fea_norm = tf.norm(fea, axis=-1, keepdims=True)
    return tf.div(fea, tf.maximum(fea_norm, 1e-12))

  def build_predict_graph(self):
    capsule_layer = CapsuleLayer(self._model_config.capsule_config,
                                 self._is_training)

    time_id_fea = [
        x[0] for x in self._hist_seq_features if 'time_id_embedding/' in x[0].name
    ]
    time_id_fea = time_id_fea[0] if len(time_id_fea) > 0 else None

    hist_seq_feas = [
        x[0] for x in self._hist_seq_features if 'time_id_embedding/' not in x[0].name
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
    simi = tf.einsum('bhe,be->bh', user_features, item_feature)
    simi = tf.pow(simi, self._model_config.simi_pow)
    simi_mask = tf.sequence_mask(num_high_capsules,
                                 self._model_config.capsule_config.max_k)

    user_features = user_features * tf.to_float(simi_mask[:, :, None])
    self._prediction_dict['user_features'] = user_features

    max_thresh = (tf.cast(simi_mask, tf.float32) * 2 - 1) * 1e32
    simi = tf.minimum(simi, max_thresh)
    simi = tf.nn.softmax(simi, axis=1)
    simi = tf.stop_gradient(simi)
    user_tower_emb = tf.einsum('bhe,bh->be', user_features, simi)

    # calculate similarity between user_tower_emb and item_tower_emb
    item_tower_emb = item_feature
    user_item_sim = self.sim(user_tower_emb, item_tower_emb)
    sim_w = tf.get_variable(
        'sim_w',
        dtype=tf.float32,
        shape=(1, 1),
        initializer=tf.ones_initializer())
    sim_b = tf.get_variable(
        'sim_b',
        dtype=tf.float32,
        shape=(1),
        initializer=tf.zeros_initializer())
    y_pred = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
    y_pred = tf.reshape(y_pred, [-1])

    if self._loss_type == LossType.CLASSIFICATION:
      self._prediction_dict['logits'] = tf.nn.sigmoid(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.reduce_join(tf.as_string(user_features), axis=-1, separator=','),
        axis=-1,
        separator='|')
    self._prediction_dict['user_emb_num'] = num_high_capsules
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_tower_emb), axis=-1, separator=',')
    return self._prediction_dict

  def build_loss_graph(self):
    if self._loss_type == LossType.CLASSIFICATION:
      logging.info('log loss is used')
      loss = losses.log_loss(self._labels[0], self._prediction_dict['logits'])
      self._loss_dict['cross_entropy_loss'] = loss
    elif self._loss_type == LossType.L2_LOSS:
      logging.info('l2 loss is used')
      loss = tf.reduce_mean(
          tf.square(self._labels[0] - self._prediction_dict['y']))
      self._loss_dict['l2_loss'] = loss
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
    return self._loss_dict

  def _build_interest_metric(self):
    user_features = self._prediction_dict['user_features']
    user_features = self.norm(user_features)
    user_feature_num = self._prediction_dict['user_emb_num']

    user_feature_sum_sqr = tf.square(tf.reduce_sum(user_features, axis=1))
    user_feature_sqr_sum = tf.reduce_sum(tf.square(user_features), axis=1)
    simi = user_feature_sum_sqr - user_feature_sqr_sum

    simi = tf.reduce_sum(
        simi, axis=1) / tf.maximum(
            tf.to_float(user_feature_num * (user_feature_num - 1)), 1.0)
    user_feature_num = tf.reduce_sum(tf.to_float(user_feature_num > 1))
    return metrics.mean(tf.reduce_sum(simi) / tf.maximum(user_feature_num, 1.0))

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['auc'] = metrics.auc(self._labels[0],
                                         self._prediction_dict['logits'])
      elif metric.WhichOneof('metric') == 'mean_absolute_error':
        assert self._loss_type == LossType.L2_LOSS
        metric_dict['mean_absolute_error'] = metrics.mean_absolute_error(
            self._labels[0], self._prediction_dict['y'])
    metric_dict['interest_similarity'] = self._build_interest_metric()
    return metric_dict

  def get_outputs(self):
    if self._loss_type == LossType.CLASSIFICATION:
      return ['logits', 'user_emb', 'item_emb']
    elif self._loss_type == LossType.L2_LOSS:
      return ['y', 'user_emb', 'item_emb']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
