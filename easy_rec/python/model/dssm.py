# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.builders import loss_builder
from easy_rec.python.layers import dnn
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.dssm_pb2 import DSSM as DSSMConfig
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.simi_pb2 import Similarity
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses
metrics = tf.metrics


class DSSM(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DSSM, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class
    assert self._model_config.WhichOneof('model') == 'dssm', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dssm
    assert isinstance(self._model_config, DSSMConfig)

    if self._loss_type == LossType.CLASSIFICATION:
      assert self._num_class == 1

    # copy_obj so that any modification will not affect original config
    self.user_tower = copy_obj(self._model_config.user_tower)
    self.user_tower_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.user_id = self.user_tower.id
    # copy_obj so that any modification will not affect original config
    self.item_tower = copy_obj(self._model_config.item_tower)
    self.item_tower_feature, _ = self._input_layer(self._feature_dict, 'item')
    self.item_id = self.item_tower.id

    if self._loss_type in [LossType.CLASSIFICATION, LossType.L2_LOSS]:
      self._is_point_wise = True
      logging.info('Use point wise dssm.')
    else:
      self._is_point_wise = False
      logging.info('Use list wise dssm.')

  def _list_wise_sim(self, user_emb, item_emb):
    batch_size = tf.shape(user_emb)[0]
    hard_neg_indices = self._feature_dict.get('hard_neg_indices', None)

    if hard_neg_indices is not None:
      tf.logging.info('With hard negative examples')
      noclk_size = tf.shape(hard_neg_indices)[0]
      pos_item_emb, neg_item_emb, hard_neg_item_emb = tf.split(
          item_emb, [batch_size, -1, noclk_size], axis=0)
    else:
      pos_item_emb = item_emb[:batch_size]
      neg_item_emb = item_emb[batch_size:]

    pos_user_item_sim = tf.reduce_sum(
        tf.multiply(user_emb, pos_item_emb), axis=1, keep_dims=True)
    neg_user_item_sim = tf.matmul(user_emb, tf.transpose(neg_item_emb))

    if hard_neg_indices is not None:
      user_emb_expand = tf.gather(user_emb, hard_neg_indices[:, 0])
      hard_neg_user_item_sim = tf.reduce_sum(
          tf.multiply(user_emb_expand, hard_neg_item_emb), axis=1)
      # scatter hard negatives sim update neg_user_item_sim
      neg_sim_shape = tf.shape(neg_user_item_sim, out_type=tf.int64)
      hard_neg_mask = tf.scatter_nd(
          hard_neg_indices,
          tf.ones_like(hard_neg_user_item_sim, dtype=tf.bool),
          shape=neg_sim_shape)
      hard_neg_user_item_sim = tf.scatter_nd(
          hard_neg_indices, hard_neg_user_item_sim, shape=neg_sim_shape)
      neg_user_item_sim = tf.where(
          hard_neg_mask, x=hard_neg_user_item_sim, y=neg_user_item_sim)

    user_item_sim = tf.concat([pos_user_item_sim, neg_user_item_sim], axis=1)
    return user_item_sim

  def _point_wise_sim(self, user_emb, item_emb):
    user_item_sim = tf.reduce_sum(
        tf.multiply(user_emb, item_emb), axis=1, keep_dims=True)
    return user_item_sim

  def sim(self, user_emb, item_emb):
    if self._is_point_wise:
      return self._point_wise_sim(user_emb, item_emb)
    else:
      return self._list_wise_sim(user_emb, item_emb)

  def norm(self, fea):
    fea_norm = tf.nn.l2_normalize(fea, axis=1)
    return fea_norm

  def build_predict_graph(self):
    num_user_dnn_layer = len(self.user_tower.dnn.hidden_units)
    last_user_hidden = self.user_tower.dnn.hidden_units.pop()
    user_dnn = dnn.DNN(self.user_tower.dnn, self._l2_reg, 'user_dnn',
                       self._is_training)
    user_tower_emb = user_dnn(self.user_tower_feature)
    user_tower_emb = tf.layers.dense(
        inputs=user_tower_emb,
        units=last_user_hidden,
        kernel_regularizer=self._l2_reg,
        name='user_dnn/dnn_%d' % (num_user_dnn_layer - 1))

    num_item_dnn_layer = len(self.item_tower.dnn.hidden_units)
    last_item_hidden = self.item_tower.dnn.hidden_units.pop()
    item_dnn = dnn.DNN(self.item_tower.dnn, self._l2_reg, 'item_dnn',
                       self._is_training)
    item_tower_emb = item_dnn(self.item_tower_feature)
    item_tower_emb = tf.layers.dense(
        inputs=item_tower_emb,
        units=last_item_hidden,
        kernel_regularizer=self._l2_reg,
        name='item_dnn/dnn_%d' % (num_item_dnn_layer - 1))

    if self._loss_type == LossType.CLASSIFICATION:
      if self._model_config.simi_func == Similarity.COSINE:
        user_tower_emb = self.norm(user_tower_emb)
        item_tower_emb = self.norm(item_tower_emb)

    user_item_sim = self.sim(user_tower_emb, item_tower_emb)
    y_pred = user_item_sim
    if self._model_config.scale_simi:
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
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.softmax(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.as_string(user_tower_emb), axis=-1, separator=',')
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_tower_emb), axis=-1, separator=',')
    return self._prediction_dict

  def build_loss_graph(self):
    if self._is_point_wise:
      return self._build_point_wise_loss_graph()
    else:
      return self._build_list_wise_loss_graph()

  def _build_list_wise_loss_graph(self):
    if self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      hit_prob = self._prediction_dict['probs'][:, :1]
      self._loss_dict['cross_entropy_loss'] = -tf.reduce_mean(
          tf.log(hit_prob + 1e-12))
      logging.info('softmax cross entropy loss is used')
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
    return self._loss_dict

  def _build_point_wise_loss_graph(self):
    label = list(self._labels.values())[0]
    if self._loss_type == LossType.CLASSIFICATION:
      pred = self._prediction_dict['logits']
      loss_name = 'cross_entropy_loss'
    elif self._loss_type == LossType.L2_LOSS:
      pred = self._prediction_dict['y']
      loss_name = 'l2_loss'
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))

    self._loss_dict[loss_name] = loss_builder.build(
        self._loss_type,
        label=label,
        pred=pred,
        loss_weight=self._sample_weight)

    # build kd loss
    kd_loss_dict = loss_builder.build_kd_loss(self.kd, self._prediction_dict,
                                              self._labels)
    self._loss_dict.update(kd_loss_dict)
    return self._loss_dict

  def build_metric_graph(self, eval_config):
    if self._is_point_wise:
      return self._build_point_wise_metric_graph(eval_config)
    else:
      return self._build_list_wise_metric_graph(eval_config)

  def _build_list_wise_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'recall_at_topk':
        logits = self._prediction_dict['logits']
        label = tf.zeros_like(logits[:, :1], dtype=tf.int64)
        metric_dict['recall_at_top%d' %
                    metric.recall_at_topk.topk] = metrics.recall_at_k(
                        label, logits, metric.recall_at_topk.topk)
      else:
        ValueError('invalid metric type: %s' % str(metric))
    return metric_dict

  def _build_point_wise_metric_graph(self, eval_config):
    metric_dict = {}
    label = list(self._labels.values())[0]
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['auc'] = metrics.auc(label, self._prediction_dict['probs'])
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['recall_at_topk%d' %
                    metric.recall_at_topk.topk] = metrics.recall_at_k(
                        label, self._prediction_dict['probs'],
                        metric.recall_at_topk.topk)
      elif metric.WhichOneof('metric') == 'mean_absolute_error':
        assert self._loss_type == LossType.L2_LOSS
        metric_dict['mean_absolute_error'] = metrics.mean_absolute_error(
            label, self._prediction_dict['y'])
      elif metric.WhichOneof('metric') == 'accuracy':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['accuracy'] = metrics.accuracy(
            label, self._prediction_dict['probs'])
      else:
        ValueError('invalid metric type: %s' % str(metric))
    return metric_dict

  def get_outputs(self):
    if self._loss_type in (LossType.CLASSIFICATION,
                           LossType.SOFTMAX_CROSS_ENTROPY):
      return ['logits', 'probs', 'user_emb', 'item_emb']
    elif self._loss_type == LossType.L2_LOSS:
      return ['y', 'user_emb', 'item_emb']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
