# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import logging

import tensorflow as tf

from easy_rec.python.core import losses as loss_lib
from easy_rec.python.core import metrics as metrics_lib
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossReduction
from easy_rec.python.protos.loss_pb2 import LossType

if tf.__version__ >= '2.0':
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
  tf = tf.compat.v1
else:
  losses = tf.losses
  metrics = tf.metrics


class SingleStageModel(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(SingleStageModel, self).__init__(model_config, feature_configs,
                                           features, labels, is_training)
    self._loss_type = self._model_config.loss_type
    self._loss_reduction = self._model_config.loss_reduction
    self._num_class = self._model_config.num_class

    if self._labels is not None:
      self._labels = list(self._labels.values())
      if self._is_classification:
        self._labels[0] = tf.cast(self._labels[0], tf.int64)
      elif self._loss_type == LossType.L2_LOSS:
        self._labels[0] = tf.cast(self._labels[0], tf.float32)

    if self._is_classification:
      assert self._num_class == 1

  def _add_to_prediction_dict(self, y_pred):
    if self._loss_type == LossType.CLASSIFICATION:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.softmax(y_pred)
    elif self._loss_type == LossType.FOCAL_LOSS:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

  def build_loss_graph(self):
    if self._loss_type == LossType.CLASSIFICATION:
      logging.info('log loss is used')
      logits = self._prediction_dict['logits']
      # yapf: disable
      label = tf.concat([
          self._labels[0][:, tf.newaxis],
          tf.zeros_like(logits, dtype=tf.int64)[:, 1:]
      ], axis=1)
      # yapf: enable
      if self._loss_reduction == LossReduction.MEAN:
        loss = losses.sigmoid_cross_entropy(label, logits)
      elif self._loss_reduction == LossReduction.MEAN_BY_BATCH_SIZE:
        batch_size = tf.to_float(tf.shape(logits)[0])
        loss = losses.sigmoid_cross_entropy(
            label, logits, reduction=losses.Reduction.SUM) / batch_size
      else:
        raise NotImplementedError
      self._loss_dict['cross_entropy_loss'] = loss
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      logging.info('softmax cross entropy loss is used')
      hit_prob = self._prediction_dict['probs'][:, :1]
      self._loss_dict['cross_entropy_loss'] = -tf.reduce_mean(
          tf.log(hit_prob + 1e-12))
    elif self._loss_type == LossType.FOCAL_LOSS:
      logging.info('focal loss is used')
      logits = self._prediction_dict['logits']
      # yapf: disable
      label = tf.concat([
          self._labels[0][:, tf.newaxis],
          tf.zeros_like(logits, dtype=tf.int64)[:, 1:]
      ], axis=1)
      # yapf: enable
      self._loss_dict[
          'cross_entropy_loss'] = loss_lib.sigmoid_focal_cross_entropy(
              label, logits, reduction=self._loss_reduction)
    elif self._loss_type == LossType.L2_LOSS:
      logging.info('l2 loss is used')
      loss = tf.reduce_mean(
          tf.square(self._labels[0] - self._prediction_dict['y']))
      self._loss_dict['l2_loss'] = loss
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
    return self._loss_dict

  @property
  def _is_classification(self):
    return (self._loss_type == LossType.CLASSIFICATION or
            self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY or
            self._loss_type == LossType.FOCAL_LOSS)

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._is_classification
        probs = self._prediction_dict['probs']
        # yapf: disable
        label = tf.concat([
            self._labels[0][:, tf.newaxis],
            tf.zeros_like(probs, dtype=tf.int64)[:, 1:]
        ], axis=1)
        # yapf: enable
        metric_dict['auc'] = metrics.auc(label, probs)
      elif metric.WhichOneof('metric') == 'gauc':
        assert self._is_classification
        probs = self._prediction_dict['probs']
        # yapf: disable
        label = tf.concat([
            self._labels[0][:, tf.newaxis],
            tf.zeros_like(probs, dtype=tf.int64)[:, 1:]
        ], axis=1)
        # yapf: enable
        metric_dict['gauc'] = metrics_lib.gauc(
            label,
            probs,
            uids=self._feature_dict[metric.gauc.uid_field],
            reduction=metric.gauc.reduction)
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._is_classification
        logits = self._prediction_dict['logits']
        label = tf.zeros_like(logits[:, :1], dtype=tf.int64)
        metric_dict['recall_at_top%d' %
                    metric.recall_at_topk.topk] = metrics.recall_at_k(
                        label, logits, metric.recall_at_topk.topk)
      elif metric.WhichOneof('metric') == 'mean_absolute_error':
        assert self._loss_type == LossType.L2_LOSS
        metric_dict['mean_absolute_error'] = metrics.mean_absolute_error(
            self._labels[0], self._prediction_dict['y'])
      elif metric.WhichOneof('metric') == 'accuracy':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['accuracy'] = metrics.accuracy(
            self._labels[0], self._prediction_dict['logits'])
    return metric_dict

  def get_outputs(self):
    if self._is_classification:
      return ['probs', 'logits']
    elif self._loss_type == LossType.L2_LOSS:
      return ['y']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
