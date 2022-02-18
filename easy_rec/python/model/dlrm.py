# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.dlrm_pb2 import DLRM as DLRMConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DLRM(RankModel):
  """Implements Deep Learning Recommendation Model for Personalization and Recommendation Systems(FaceBook)."""

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DLRM, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert model_config.WhichOneof('model') == 'dlrm', \
        'invalid model config: %s' % model_config.WhichOneof('model')
    self._model_config = model_config.dlrm
    assert isinstance(self._model_config, DLRMConfig)
    assert self._input_layer.has_group(
        'sparse'), 'sparse group is not specified'
    _, self._sparse_features = self._input_layer(self._feature_dict, 'sparse')
    assert self._input_layer.has_group('dense'), 'dense group is not specified'
    self._dense_feature, _ = self._input_layer(self._feature_dict, 'dense')

  def build_predict_graph(self):
    bot_dnn = dnn.DNN(self._model_config.bot_dnn, self._l2_reg, 'bot_dnn',
                      self._is_training)
    dense_fea = bot_dnn(self._dense_feature)
    logging.info('arch_interaction_op = %s' %
                 self._model_config.arch_interaction_op)
    if self._model_config.arch_interaction_op == 'cat':
      all_fea = tf.concat([dense_fea] + self._sparse_features, axis=1)
    elif self._model_config.arch_interaction_op == 'dot':
      assert dense_fea.get_shape()[1] == self._sparse_features[0].get_shape()[1], \
          'bot_dnn last hidden[%d] != sparse feature embedding_dim[%d]' % (
          dense_fea.get_shape()[1], self._sparse_features[0].get_shape()[1])

      all_feas = [dense_fea] + self._sparse_features
      all_feas = [x[:, None, :] for x in all_feas]
      all_feas = tf.concat(all_feas, axis=1)
      num_fea = all_feas.get_shape()[1]
      interaction = tf.einsum('bne,bme->bnm', all_feas, all_feas)
      offset = 0 if self._model_config.arch_interaction_itself else 1
      upper_tri = []
      for i in range(num_fea):
        upper_tri.append(interaction[:, i, (i + offset):num_fea])
      upper_tri = tf.concat(upper_tri, axis=1)
      concat_feas = [upper_tri] + self._sparse_features
      if self._model_config.arch_with_dense_feature:
        concat_feas.append(dense_fea)
      all_fea = tf.concat(concat_feas, axis=1)

    top_dnn = dnn.DNN(self._model_config.top_dnn, self._l2_reg, 'top_dnn',
                      self._is_training)
    all_fea = top_dnn(all_fea)
    logits = tf.layers.dense(
        all_fea, 1, kernel_regularizer=self._l2_reg, name='output')

    self._add_to_prediction_dict(logits)

    return self._prediction_dict
