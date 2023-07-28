# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.easy_rec_model import EasyRecModel

from easy_rec.python.protos.ultragcn_pb2 import ULTRAGCN as ULTRAGCNConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class ULTRAGCN(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(ULTRAGCN, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    self._model_config = model_config.ultragcn
    assert isinstance(self._model_config, ULTRAGCNConfig)
    self._user_num = self._model_config.user_num
    self._item_num = self._model_config.item_num
    self._emb_dim = self._model_config.output_dim
    self._i2i_weight = self._model_config.i2i_weight
    self._neg_weight = self._model_config.neg_weight
    self._l2_weight = self._model_config.l2_weight
    self._user_emb = None
    self._item_emb = None

    if features.get('features') is not None:
      self._user_ids = features.get('features')[0]
      self._user_degrees = features.get('features')[1]
      self._item_ids = features.get('features')[2]
      self._item_degrees = features.get('features')[3]
      self._nbr_ids = features.get('features')[4]
      self._nbr_weights = features.get('features')[5]
      self._neg_ids = features.get('features')[6]
    else:
      self._user_ids = features.get('id')
      self._user_degrees = None
      self._item_ids = features.get('id')
      self._item_degrees = None
      self._nbr_ids = features.get('id')
      self._nbr_weights = None
      self._neg_ids = features.get('id')

    _user_emb = tf.get_variable("user_emb",
                                [self._user_num, self._emb_dim],
                                trainable=True)
    _item_emb = tf.get_variable("item_emb",
                                [self._item_num, self._emb_dim],
                                trainable=True)

    self._user_emb = tf.convert_to_tensor(_user_emb)
    self._item_emb = tf.convert_to_tensor(_item_emb)

  def build_predict_graph(self):
    user_emb = tf.nn.embedding_lookup(self._user_emb, self._user_ids)
    item_emb = tf.nn.embedding_lookup(self._item_emb, self._item_ids)
    nbr_emb = tf.nn.embedding_lookup(self._item_emb, self._nbr_ids)
    neg_emb = tf.nn.embedding_lookup(self._item_emb, self._neg_ids)
    self._prediction_dict['user_emb'] = user_emb
    self._prediction_dict['item_emb'] = item_emb
    self._prediction_dict['nbr_emb'] = nbr_emb
    self._prediction_dict['neg_emb'] = neg_emb
    self._prediction_dict['user_embedding'] = tf.reduce_join(
        tf.as_string(user_emb), axis=-1, separator=',')
    self._prediction_dict['item_embedding'] = tf.reduce_join(
        tf.as_string(item_emb), axis=-1, separator=',')

    return self._prediction_dict

  def build_loss_graph(self):
    # UltraGCN base u2i
    pos_logit = tf.reduce_sum(self._prediction_dict['user_emb'] * self._prediction_dict['item_emb'], axis=-1)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(pos_logit), logits=pos_logit)
    neg_logit = tf.reduce_sum(tf.expand_dims(self._prediction_dict['user_emb'], axis=1) * self._prediction_dict['neg_emb'], axis=-1)
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_logit), logits=neg_logit)
    loss_u2i = tf.reduce_sum(true_xent * (1 + 1 / tf.sqrt(self._user_degrees * self._item_degrees))) \
      + self._neg_weight * tf.reduce_sum(tf.reduce_mean(negative_xent, axis=-1))
    # UltraGCN i2i
    nbr_logit = tf.reduce_sum(tf.expand_dims(self._prediction_dict['user_emb'], axis=1) * self._prediction_dict['nbr_emb'], axis=-1) # [batch_size, nbr_num]
    nbr_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(nbr_logit), logits=nbr_logit)  
    loss_i2i = tf.reduce_sum(nbr_xent * (1 + self._nbr_weights))
    # regularization
    loss_l2 = tf.nn.l2_loss(self._prediction_dict['user_emb']) + tf.nn.l2_loss(self._prediction_dict['item_emb']) +\
      tf.nn.l2_loss(self._prediction_dict['nbr_emb']) + tf.nn.l2_loss(self._prediction_dict['neg_emb'])
    
    loss = loss_u2i + self._i2i_weight * loss_i2i + self._l2_weight * loss_l2
    return {'cross_entropy': loss}

  def build_metric_graph(self, eval_config):
    return {}

  def get_outputs(self):
    # emb_1 = tf.reduce_join(tf.as_string(self._prediction_dict['user_embedding']), axis=-1, separator=',')
    # emb_2 = tf.reduce_join(tf.as_string(self._prediction_dict['item_embedding'] ), axis=-1, separator=',')
    return ['user_embedding','item_embedding']
  
  
  def build_metric_graph(self, eval_config):
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