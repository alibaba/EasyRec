# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.dssm_pb2 import DSSM as DSSMConfig
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
  tf = tf.compat.v1
else:
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

    if self._labels is not None:
      self._labels = list(self._labels.values())
      if self._loss_type == LossType.CLASSIFICATION:
        if tf.__version__ >= '2.0':
          self._labels[0] = tf.cast(self._labels[0], tf.int64)
        else:
          self._labels[0] = tf.to_int64(self._labels[0])
      elif self._loss_type == LossType.L2_LOSS:
        if tf.__version__ >= '2.0':
          self._labels[0] = tf.cast(self._labels[0], tf.float32)
        else:
          self._labels[0] = tf.to_float(self._labels[0])

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

    regularizers.apply_regularization(
        self._emb_reg, weights_list=[self.user_tower_feature])
    regularizers.apply_regularization(
        self._emb_reg, weights_list=[self.item_tower_feature])

    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def sim(self, user_emb, item_emb):
    user_item_sim = tf.reduce_sum(
        tf.multiply(user_emb, item_emb), axis=1, keep_dims=True)
    return user_item_sim

  def norm(self, fea):
    fea_norm = tf.norm(fea, axis=1, keepdims=True)
    return tf.div(fea, tf.maximum(fea_norm, 1e-12))

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
      user_tower_emb = self.norm(user_tower_emb)
      item_tower_emb = self.norm(item_tower_emb)

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
        tf.as_string(user_tower_emb), axis=-1, separator=',')
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

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['auc'] = metrics.auc(self._labels[0],
                                         self._prediction_dict['logits'])
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['recall_at_topk'] = metrics.recall_at_k(
            self._labels[0], self._prediction_dict['logits'],
            metric.recall_at_topk.topk)
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
    if self._loss_type == LossType.CLASSIFICATION:
      return ['logits', 'user_emb', 'item_emb']
    elif self._loss_type == LossType.L2_LOSS:
      return ['y', 'user_emb', 'item_emb']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
