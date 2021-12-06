# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.builders import loss_builder
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses
metrics = tf.metrics


class MatchModel(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MatchModel, self).__init__(model_config, feature_configs, features,
                                     labels, is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class

    if self._loss_type == LossType.CLASSIFICATION:
      assert self._num_class == 1

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
          tf.ones_like(hard_neg_user_item_sim, dtype=tf.float32),
          shape=neg_sim_shape)
      # set tail positions to -1e32, so that after exp(x), will be zero
      hard_neg_mask = (1 - hard_neg_mask) * (-1e32)
      hard_neg_user_item_sim = tf.scatter_nd(
          hard_neg_indices, hard_neg_user_item_sim,
          shape=neg_sim_shape) * hard_neg_mask

    user_item_sim = [pos_user_item_sim, neg_user_item_sim]
    if hard_neg_indices is not None:
      user_item_sim.append(hard_neg_user_item_sim)
    return tf.concat(user_item_sim, axis=1)

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
    fea_norm = tf.nn.l2_normalize(fea, axis=-1)
    return fea_norm

  def build_predict_graph(self):
    raise NotImplementedError('MatchModel could not be instantiated')

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
