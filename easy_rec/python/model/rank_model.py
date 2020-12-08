# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class RankModel(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(RankModel, self).__init__(model_config, feature_configs, features,
                                    labels, is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class

    if self._labels is not None:
      self._label_name = list(self._labels.keys())[0]

  def _output_to_prediction_impl(self,
                                 output,
                                 loss_type,
                                 num_class=1,
                                 suffix=''):
    prediction_dict = {}
    if loss_type == LossType.CLASSIFICATION:
      if num_class == 1:
        output = tf.squeeze(output, axis=1)
        probs = tf.sigmoid(output)
        prediction_dict['logits' + suffix] = output
        prediction_dict['probs' + suffix] = probs
      else:
        probs = tf.nn.softmax(output, axis=1)
        prediction_dict['logits' + suffix] = output
        prediction_dict['probs' + suffix] = probs
        prediction_dict['y' + suffix] = tf.argmax(output, axis=1)
    else:
      output = tf.squeeze(output, axis=1)
      prediction_dict['y' + suffix] = output
    return prediction_dict

  def _add_to_prediction_dict(self, output):
    self._prediction_dict.update(
        self._output_to_prediction_impl(
            output, loss_type=self._loss_type, num_class=self._num_class))

  def _build_loss_impl(self,
                       loss_type,
                       label_name,
                       loss_weight=1.0,
                       num_class=1,
                       suffix=''):
    loss_dict = {}
    if loss_type == LossType.CLASSIFICATION:
      if num_class == 1:
        loss = tf.losses.sigmoid_cross_entropy(
            self._labels[label_name],
            logits=self._prediction_dict['logits' + suffix],
            weights=loss_weight)
        loss_dict['cross_entropy_loss' + suffix] = loss
      else:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self._labels[label_name],
            logits=self._prediction_dict['logits' + suffix],
            weights=loss_weight)
        loss_dict['cross_entropy_loss' + suffix] = loss
    elif loss_type == LossType.L2_LOSS:
      logging.info('l2 loss is used')
      loss = tf.losses.mean_squared_error(
          labels=self._labels[label_name],
          predictions=self._prediction_dict['y' + suffix],
          weights=loss_weight)
      loss_dict['l2_loss' + suffix] = loss
    else:
      raise ValueError('invalid loss type: %s' % str(loss_type))
    return loss_dict

  def build_loss_graph(self):
    self._loss_dict.update(
        self._build_loss_impl(
            self._loss_type,
            label_name=self._label_name,
            num_class=self._num_class))
    return self._loss_dict

  def _build_metric_impl(self,
                         metric,
                         loss_type,
                         label_name,
                         num_class=1,
                         suffix=''):
    metric_dict = {}
    if metric.WhichOneof('metric') == 'auc':
      assert loss_type == LossType.CLASSIFICATION
      assert num_class == 1
      label = tf.to_int64(self._labels[label_name])
      metric_dict['auc' + suffix] = tf.metrics.auc(
          label, self._prediction_dict['probs' + suffix])
    elif metric.WhichOneof('metric') == 'recall_at_topk':
      assert loss_type == LossType.CLASSIFICATION
      assert num_class > 1
      label = tf.to_int64(self._labels[label_name])
      metric_dict['recall_at_topk' + suffix] = tf.metrics.recall_at_k(
          label, self._prediction_dict['logits' + suffix],
          metric.recall_at_topk.topk)
    elif metric.WhichOneof('metric') == 'mean_absolute_error':
      label = tf.to_float(self._labels[label_name])
      if loss_type == LossType.L2_LOSS:
        metric_dict['mean_absolute_error' +
                    suffix] = tf.metrics.mean_absolute_error(
                        label, self._prediction_dict['y' + suffix])
      elif loss_type == LossType.CLASSIFICATION and num_class == 1:
        metric_dict['mean_absolute_error' +
                    suffix] = tf.metrics.mean_absolute_error(
                        label, self._prediction_dict['probs' + suffix])
      else:
        assert False, 'mean_absolute_error is not supported for this model'
    elif metric.WhichOneof('metric') == 'mean_squared_error':
      label = tf.to_float(self._labels[label_name])
      if loss_type == LossType.L2_LOSS:
        metric_dict['mean_squared_error' +
                    suffix] = tf.metrics.mean_squared_error(
                        label, self._prediction_dict['y' + suffix])
      elif loss_type == LossType.CLASSIFICATION and num_class == 1:
        metric_dict['mean_squared_error' +
                    suffix] = tf.metrics.mean_squared_error(
                        label, self._prediction_dict['probs' + suffix])
      else:
        assert False, 'mean_squared_error is not supported for this model'
    elif metric.WhichOneof('metric') == 'accuracy':
      assert loss_type == LossType.CLASSIFICATION
      assert num_class > 1
      label = tf.to_int64(self._labels[label_name])
      metric_dict['accuracy' + suffix] = tf.metrics.accuracy(
          label, self._prediction_dict['y' + suffix])
    return metric_dict

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=self._loss_type,
              label_name=self._label_name,
              num_class=self._num_class))
    return metric_dict

  def _get_outputs_impl(self, loss_type, num_class=1, suffix=''):
    if loss_type == LossType.CLASSIFICATION:
      if num_class == 1:
        return ['probs' + suffix, 'logits' + suffix]
      else:
        return ['y' + suffix, 'probs' + suffix, 'logits' + suffix]
    elif loss_type == LossType.L2_LOSS:
      return ['y' + suffix]
    else:
      raise ValueError('invalid loss type: %s' % str(loss_type))

  def get_outputs(self):
    return self._get_outputs_impl(self._loss_type, self._num_class)
