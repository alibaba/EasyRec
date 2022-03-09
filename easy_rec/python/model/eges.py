# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.easy_rec_model import EasyRecModel

from easy_rec.python.protos.eges_pb2 import EGES as EGESConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class EGES(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(EGES, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    self._model_config = model_config.eges
    assert isinstance(self._model_config, EGESConfig)
    self._group_name = 'item'
    assert self._input_layer.has_group(
        self._group_name), 'group[%s] is not specified' % self._group_name

    if labels is not None:
      self._positive_features = labels['positive_fea']
      self._negative_features = labels['negative_fea']
      self._src_features = features
      self._positive_embedding, _ = self._input_layer(self._positive_features,
                                                      self._group_name)
      self._negative_embedding, _ = self._input_layer(self._negative_features,
                                                      self._group_name)
      self._src_embedding, _ = self._input_layer(self._src_features,
                                                 self._group_name)
    else:
      self._src_embedding, _ = self._input_layer(features, self._group_name)
      self._negative_embedding = None
      self._positive_embedding = None

  def build_predict_graph(self):
    if self._negative_embedding is None:
      logging.info('build predict item embedding graph.')
      src_embedding = tf.layers.batch_normalization(
          self._src_embedding,
          training=self._is_training,
          trainable=True,
          name='%s_fea_bn' % self._group_name)
      dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
                          self._is_training)
      src_embedding = dnn_layer(src_embedding)
      self._prediction_dict['item_embedding'] = src_embedding
      return self._prediction_dict

    all_embedding = tf.concat([
        self._src_embedding, self._positive_embedding, self._negative_embedding
    ],
                              axis=0)  # noqa: E126
    all_embedding = tf.layers.batch_normalization(
        all_embedding,
        training=self._is_training,
        trainable=True,
        name='%s_fea_bn' % self._group_name)
    batch_size = tf.shape(self._src_embedding)[0]
    src_embedding = all_embedding[:batch_size]
    pos_embedding = all_embedding[batch_size:(batch_size * 2)]
    neg_embedding = all_embedding[(batch_size * 2):]
    tf.summary.scalar('actual_batch_size', tf.shape(src_embedding)[0])
    tf.summary.histogram('src_fea', src_embedding)
    tf.summary.histogram('neg_fea', neg_embedding)
    tf.summary.histogram('pos_fea', pos_embedding)
    dnn_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'dnn',
                        self._is_training)
    all_embedding = dnn_layer(all_embedding)

    embed_dim = all_embedding.get_shape()[-1]
    src_embedding = all_embedding[:batch_size]
    pos_embedding = all_embedding[batch_size:(batch_size * 2)]
    neg_embedding = all_embedding[(batch_size * 2):]
    neg_embedding = tf.reshape(neg_embedding, [batch_size, -1, embed_dim])
    target_embedding = tf.concat([pos_embedding[:, None, :], neg_embedding],
                                 axis=1)

    self._prediction_dict['item_embedding'] = src_embedding
    self._prediction_dict['target_embedding'] = target_embedding
    return self._prediction_dict

  def build_loss_graph(self):
    src_embedding = self._prediction_dict['item_embedding']
    target_embedding = self._prediction_dict['target_embedding']
    logits = tf.einsum('be,bne->bn', src_embedding, target_embedding)
    batch_size = tf.shape(src_embedding)[0]
    labels = tf.zeros([batch_size], dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return {'cross_entropy': tf.reduce_mean(loss)}

  def build_metric_graph(self, eval_config):
    return {}

  def get_outputs(self):
    return ['item_embedding']
