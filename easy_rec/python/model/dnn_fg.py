# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.core import metrics as metrics_lib
from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos.dnnfg_pb2 import DNNFG as DNNFGConfig  # NOQA

if tf.__version__ >= '2.0':
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
  tf = tf.compat.v1
else:
  losses = tf.losses
  metrics = tf.metrics

class DNNFG(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DNNFG, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'dnnfg', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dnnfg
    assert isinstance(self._model_config, DNNFGConfig)

    self.feature, _ = self._input_layer(self._feature_dict, 'all')

    if self._labels is not None:
      self._labels = list(self._labels.values())
      self._labels[0] = tf.cast(self._labels[0], tf.int64)
    
    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def build_predict_graph(self):
    if self._mode != tf.estimator.ModeKeys.PREDICT:
      assert 'hard_neg_indices' not in self._feature_dict
      num_neg = self._feature_dict['__num_neg_sample__']
      all_fea = tf.reshape(self.feature, [-1, 1 + num_neg, self.feature.shape[-1]])
    else:
      all_fea = self.feature

    dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
                        self._is_training)
    all_fea = dnn_layer(all_fea)
    output = tf.layers.dense(all_fea, 1, name='output')
    output = tf.squeeze(output, axis=-1)

    self._add_to_prediction_dict(output)

    return self._prediction_dict

  def _add_to_prediction_dict(self, y_pred):
    self._prediction_dict['logits'] = y_pred
    self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)

  def build_loss_graph(self):
    logits = self._prediction_dict['logits']
    label = tf.to_float(self._labels[0])
    self._loss_dict['sigmoid_cross_entropy_loss'] = \
            self._model_config.pointwise_loss_weight * tf.losses.sigmoid_cross_entropy(
                label, logits=logits[:, 0])

    return self._loss_dict

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        probs = self._prediction_dict['probs']
        metric_dict['auc'] = metrics.auc(self._labels[0], probs[:, 0])
      elif metric.WhichOneof('metric') == 'gauc':
        probs = self._prediction_dict['probs']
        metric_dict['gauc'] = metrics_lib.gauc(
            self._labels[0],
            probs[:, 0],
            uids=self._feature_dict[metric.gauc.uid_field],
            reduction=metric.gauc.reduction)
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        mask = tf.equal(self._labels[0], 1)
        logits = tf.boolean_mask(self._prediction_dict['logits'], mask)
        label = tf.zeros_like(logits[:, :1], dtype=tf.int64)
        with tf.device('/cpu:0'):
          metric_dict['recall_at_top%d' %
                      metric.recall_at_topk.topk] = metrics.recall_at_k(
                          label, logits, metric.recall_at_topk.topk)
    return metric_dict

  def get_outputs(self):
      outputs = super(DNNFG, self).get_outputs()
      return outputs