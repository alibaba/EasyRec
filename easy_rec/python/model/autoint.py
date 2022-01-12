# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import multihead_attention
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.autoint_pb2 import AutoInt as AutoIntConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class AutoInt(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(AutoInt, self).__init__(model_config, feature_configs, features,
                                  labels, is_training)
    assert self._model_config.WhichOneof('model') == 'autoint', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._features, _ = self._input_layer(self._feature_dict, 'all')
    self._feature_num = len(self._model_config.feature_groups[0].feature_names)
    self._seq_key_num = 0
    if len(self._model_config.feature_groups[0].sequence_features) > 0:
      for seq_fea in self._model_config.feature_groups[0].sequence_features:
        for seq_att in seq_fea.seq_att_map:
          self._feature_num += len(seq_att.hist_seq)
          self._seq_key_num += len(seq_att.key)
    self._model_config = self._model_config.autoint
    assert isinstance(self._model_config, AutoIntConfig)

    fea_emb_dim_list = []
    for feature_config in feature_configs:
      fea_emb_dim_list.append(feature_config.embedding_dim)
    assert len(set(fea_emb_dim_list)) == 1 and len(fea_emb_dim_list) == self._feature_num, \
        'AutoInt requires that all feature dimensions must be consistent.'

    self._d_model = fea_emb_dim_list[0]
    self._head_num = self._model_config.multi_head_num
    self._head_size = self._model_config.multi_head_size

  def build_predict_graph(self):
    logging.info('feature_num: {0}'.format(self._feature_num))

    attention_fea = tf.reshape(
        self._features,
        shape=[-1, self._feature_num + self._seq_key_num, self._d_model])

    for i in range(self._model_config.interacting_layer_num):
      attention_layer = multihead_attention.MultiHeadAttention(
          head_num=self._head_num,
          head_size=self._head_size,
          l2_reg=self._l2_reg,
          use_res=True,
          name='multi_head_self_attention_layer_%d' % i)
      attention_fea = attention_layer(attention_fea)

    attention_fea = tf.reshape(
        attention_fea,
        shape=[-1, attention_fea.shape[1] * attention_fea.shape[2]])

    final = tf.layers.dense(attention_fea, self._num_class, name='output')

    self._add_to_prediction_dict(final)

    return self._prediction_dict
