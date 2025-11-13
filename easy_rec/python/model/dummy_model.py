# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.model.easy_rec_model import EasyRecModel


class DummyModel(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DummyModel, self).__init__(model_config, feature_configs, features,
                                     labels, is_training)

    if self._labels is not None:
      self._labels = list(self._labels.values())
      if self._labels[0].dtype != tf.float32:
        self._labels[0] = tf.ones_like(self._labels[0], tf.float32)

  def build_predict_graph(self):
    input_data = tf.random_uniform(tf.shape(self._labels[0]), dtype=tf.float32)
    input_data = tf.reshape(input_data, [-1, 1])
    output = tf.layers.dense(inputs=input_data, units=1, name='layer_0')
    self._prediction_dict['output'] = output
    for key in self._feature_dict:
      val = self._feature_dict[key]
      if isinstance(val, tf.sparse.SparseTensor):
        val = val.values
      self._prediction_dict[key] = val
    return self._prediction_dict

  def build_loss_graph(self):
    return {
        'cross_ent':
            tf.reduce_sum(
                tf.square(self._prediction_dict['output'] - self._labels[0]))
    }

  def get_outputs(self):
    return ['output']

  def build_metric_graph(self):
    return {}
