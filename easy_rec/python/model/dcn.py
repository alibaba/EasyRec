# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.dcn_pb2 import DCN as DCNConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DCN(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DCN, self).__init__(model_config, feature_configs, features, labels,
                              is_training)
    assert self._model_config.WhichOneof('model') == 'dcn', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dcn
    assert isinstance(self._model_config, DCNConfig)

    self._features, _ = self._input_layer(self._feature_dict, 'all')

  def _cross_net(self, tensor, num_cross_layers):
    x = x0 = tensor
    input_dim = tensor.shape[-1]
    for i in range(num_cross_layers):
      name = 'cross_layer_%s' % i
      w = tf.get_variable(
          name=name + '_w',
          dtype=tf.float32,
          shape=(input_dim),
      )
      b = tf.get_variable(name=name + '_b', dtype=tf.float32, shape=(input_dim))
      xw = tf.reduce_sum(x * w, axis=1, keepdims=True)  # (B, 1)
      x = tf.math.add(tf.math.add(x0 * xw, b), x)
    return x

  def build_predict_graph(self):
    tower_fea_arr = []
    # deep tower
    deep_tower_config = self._model_config.deep_tower

    dnn_layer = dnn.DNN(deep_tower_config.dnn, self._l2_reg, 'dnn',
                        self._is_training)
    deep_tensor = dnn_layer(self._features)
    tower_fea_arr.append(deep_tensor)
    # cross tower
    cross_tower_config = self._model_config.cross_tower
    num_cross_layers = cross_tower_config.cross_num
    cross_tensor = self._cross_net(self._features, num_cross_layers)
    tower_fea_arr.append(cross_tensor)
    # final tower
    all_fea = tf.concat(tower_fea_arr, axis=1)
    final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                              'final_dnn', self._is_training)
    all_fea = final_dnn_layer(all_fea)
    output = tf.layers.dense(all_fea, self._num_class, name='output')

    self._add_to_prediction_dict(output)

    return self._prediction_dict
