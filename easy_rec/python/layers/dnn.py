# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.utils.activation import get_activation

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DNN:

  def __init__(self,
               dnn_config,
               l2_reg,
               name='dnn',
               is_training=False,
               last_layer_no_activation=False,
               last_layer_no_batch_norm=False):
    """Initializes a `DNN` Layer.

    Args:
      dnn_config: instance of easy_rec.python.protos.dnn_pb2.DNN
      l2_reg: l2 regularizer
      name: scope of the DNN, so that the parameters could be separated from other dnns
      is_training: train phase or not, impact batch_norm and dropout
      last_layer_no_activation: in last layer, use or not use activation
      last_layer_no_batch_norm: in last layer, use or not use batch norm
    """
    self._config = dnn_config
    self._l2_reg = l2_reg
    self._name = name
    self._is_training = is_training
    logging.info('dnn activation function = %s' % self._config.activation)
    self.activation = get_activation(
        self._config.activation, training=is_training)
    self._last_layer_no_activation = last_layer_no_activation
    self._last_layer_no_batch_norm = last_layer_no_batch_norm

  @property
  def hidden_units(self):
    return self._config.hidden_units

  @property
  def dropout_ratio(self):
    return self._config.dropout_ratio

  def __call__(self, deep_fea, hidden_layer_feature_output=False):
    hidden_units_len = len(self.hidden_units)
    if hidden_units_len == 1 and self.hidden_units[0] == 0:
      return deep_fea

    hidden_feature_dict = {}
    for i, unit in enumerate(self.hidden_units):
      deep_fea = tf.layers.dense(
          inputs=deep_fea,
          units=unit,
          kernel_regularizer=self._l2_reg,
          activation=None,
          name='%s/dnn_%d' % (self._name, i))
      if self._config.use_bn and ((i + 1 < hidden_units_len) or
                                  not self._last_layer_no_batch_norm):
        deep_fea = tf.layers.batch_normalization(
            deep_fea,
            training=self._is_training,
            trainable=True,
            name='%s/dnn_%d/bn' % (self._name, i))
      if (i + 1 < hidden_units_len) or not self._last_layer_no_activation:
        deep_fea = self.activation(
            deep_fea, name='%s/dnn_%d/act' % (self._name, i))
      if len(self.dropout_ratio) > 0 and self._is_training:
        assert self.dropout_ratio[
            i] < 1, 'invalid dropout_ratio: %.3f' % self.dropout_ratio[i]
        deep_fea = tf.nn.dropout(
            deep_fea,
            keep_prob=1 - self.dropout_ratio[i],
            name='%s/%d/dropout' % (self._name, i))

      if hidden_layer_feature_output:
        hidden_feature_dict['hidden_layer' + str(i)] = deep_fea
        if (i + 1 == hidden_units_len):
          hidden_feature_dict['hidden_layer_end'] = deep_fea
          return hidden_feature_dict
    else:
      return deep_fea
