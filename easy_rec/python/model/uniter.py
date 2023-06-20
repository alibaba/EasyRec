# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import uniter
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.uniter_pb2 import Uniter as UNITERConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class Uniter(RankModel):
  """UNITER: UNiversal Image-TExt Representation Learning.

  See the original paper:
  https://arxiv.org/abs/1909.11740
  """

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(Uniter, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'uniter', (
        'invalid model config: %s' % self._model_config.WhichOneof('model'))

    self._uniter_layer = uniter.Uniter(model_config, feature_configs, features,
                                       self._model_config.uniter.config,
                                       self._input_layer)
    self._model_config = self._model_config.uniter

  def build_predict_graph(self):
    hidden = self._uniter_layer(self._is_training, l2_reg=self._l2_reg)
    final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                              'final_dnn', self._is_training)
    all_fea = final_dnn_layer(hidden)

    final = tf.layers.dense(all_fea, self._num_class, name='output')
    self._add_to_prediction_dict(final)
    return self._prediction_dict
