# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SeqInputLayer(object):

  def __init__(self, feature_configs, feature_groups_config):
    self._feature_groups_config = {
        x.group_name: x for x in feature_groups_config
    }
    wide_and_deep_dict = self.get_wide_deep_dict()
    self._fc_parser = FeatureColumnParser(feature_configs, wide_and_deep_dict)

  def __call__(self, features, group_name):
    feature_column_dict = self._fc_parser.deep_columns

    builder = feature_column._LazyBuilder(features)
    key_features = []
    hist_emb_features = []
    hist_seqlen_features = []
    feature_dict = self._feature_groups_config[group_name]

    for x in feature_dict.seq_att_map:
      with tf.variable_scope('emd_' + x.key, reuse=tf.AUTO_REUSE):
        key_tensor = feature_column_dict[x.key]._get_dense_tensor(builder)
      with tf.variable_scope('hist_emd_' + x.hist_seq, reuse=tf.AUTO_REUSE):
        hist_embedding, hist_seq_len = feature_column_dict[
            x.hist_seq]._get_sequence_dense_tensor(builder)

      key_features.append(key_tensor)
      hist_emb_features.append(hist_embedding)
      hist_seqlen_features.append(hist_seq_len)

      # din_features.append({'key': key_tensor, 'hist_seq_emb': hist_embedding, 'hist_seq_len': hist_seq_len})
    if len(feature_dict.seq_att_map) > 1:
      features = {
          'key': tf.concat(key_features, axis=1),
          'hist_seq_emb': tf.concat(hist_emb_features, axis=2),
          'hist_seq_len': hist_seqlen_features[0]
      }
    else:
      features = {
          'key': key_features[0],
          'hist_seq_emb': hist_emb_features[0],
          'hist_seq_len': hist_seqlen_features[0]
      }
    return features

  def get_wide_deep_dict(self):
    wide_and_deep_dict = {}
    for group_name_config in self._feature_groups_config.values():
      for x in group_name_config.seq_att_map:
        wide_and_deep_dict[x.key] = WideOrDeep.DEEP
        wide_and_deep_dict[x.hist_seq] = WideOrDeep.DEEP
    return wide_and_deep_dict
