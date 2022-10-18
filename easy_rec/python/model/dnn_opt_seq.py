# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.core import metrics as metrics_lib
from easy_rec.python.layers import dnn
from easy_rec.python.model.single_stage_model import SingleStageModel

from easy_rec.python.protos.dnn_opt_pb2 import DNNOptSeq as DNNOptSeqConfig

if tf.__version__ >= '2.0':
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
  tf = tf.compat.v1
else:
  losses = tf.losses
  metrics = tf.metrics

class DNNOptSeq(SingleStageModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DNNOptSeq, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'dnn_opt_seq', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dnn_opt_seq
    assert isinstance(self._model_config, DNNOptSeqConfig)

    self.user_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.item_feature, _ = self._input_layer(self._feature_dict, 'item')

    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def build_predict_graph(self):
    if self._mode != tf.estimator.ModeKeys.PREDICT:
        assert 'hard_neg_indices' not in self._feature_dict
        user_feature = self.user_feature
        batch_size = tf.shape(user_feature)[0]
        pos_item_feature = self.item_feature[:batch_size]
        neg_item_feature = self.item_feature[batch_size:]
        num_neg = tf.shape(neg_item_feature)[0]
        pos_ui_fea = tf.concat([user_feature[:, :-num_neg, :], pos_item_feature[:, tf.newaxis, :]], axis=-1)
        neg_ui_fea = tf.concat([
            user_feature[:, -num_neg:, :],
            tf.tile(
                neg_item_feature[tf.newaxis, :, :],
                multiples=[batch_size, 1, 1])
        ], axis=-1)
        all_fea = tf.concat([pos_ui_fea, neg_ui_fea], axis=1)  # [B num_neg+1 emb]
    else:
        all_fea = tf.concat([self.user_feature, self.item_feature], axis=-1)

    # dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
    #                     self._is_training,True) ## dice 
    dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
                        self._is_training) ## relu
    all_fea = dnn_layer(all_fea)
    output = tf.layers.dense(all_fea, 1, name='output')
    output = tf.squeeze(output, axis=-1)

    self._add_to_prediction_dict(output)

    return self._prediction_dict

  def build_loss_graph(self):
    logging.info('softmax cross entropy loss is used')
    logits = self._prediction_dict['logits']

    label = tf.to_float(self._labels[0])
    weight = label + (1.0 - label) * self._model_config.hard_neg_softmax_weight
    # hit_prob = tf.nn.softmax(logits)[:, 0]
    # self._loss_dict['softmax_cross_entropy_loss'] = \
    #     - self._model_config.pairwise_loss_weight * tf.reduce_sum(
    #         tf.log(hit_prob + 1e-12) * weight) / tf.reduce_sum(weight)
    self._loss_dict['softmax_cross_entropy_loss'] = \
        self._model_config.pairwise_loss_weight * tf.losses.sparse_softmax_cross_entropy(
            labels=tf.zeros_like(logits[:, 0], dtype=tf.int64),
            logits=logits,
            weights=weight,
            reduction=tf.losses.Reduction.MEAN)
    if self._model_config.pointwise_loss_weight > 0:
        self._loss_dict['sigmoid_cross_entropy_loss'] = \
            self._model_config.pointwise_loss_weight * tf.losses.sigmoid_cross_entropy(
                label, logits=logits[:, 0])

    return self._loss_dict

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._is_classification
        probs = self._prediction_dict['probs']
        # self._labels[0] = tf.Print(self._labels[0],['self._labels[0]',self._labels[0]],summarize=256)
        # probs = tf.Print(probs,['probsprobs[:, 0]',probs[:, 0]],summarize=256)
        # probs = tf.Print(probs,['probsprobs shape',tf.shape(probs)],summarize=256)
        metric_dict['auc'] = metrics.auc(self._labels[0], probs[:, 0])
      elif metric.WhichOneof('metric') == 'gauc':
        assert self._is_classification
        probs = self._prediction_dict['probs']
        metric_dict['gauc'] = metrics_lib.gauc(
            self._labels[0],
            probs[:, 0],
            uids=self._feature_dict[metric.gauc.uid_field],
            reduction=metric.gauc.reduction)
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._is_classification
        mask = tf.equal(self._labels[0], 1)
        logits = tf.boolean_mask(self._prediction_dict['logits'], mask)
        label = tf.zeros_like(logits[:, :1], dtype=tf.int64)
        with tf.device('/cpu:0'):
          metric_dict['recall_at_top%d' %
                      metric.recall_at_topk.topk] = metrics.recall_at_k(
                          label, logits, metric.recall_at_topk.topk)
    return metric_dict

  def get_outputs(self):
      outputs = super(DNNOptSeq, self).get_outputs()
      return outputs