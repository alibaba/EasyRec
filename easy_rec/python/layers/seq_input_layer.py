# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging

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
    feature_column_dict.update(self._fc_parser.sequence_columns)

    builder = feature_column._LazyBuilder(features)

    feature_dict = self._feature_groups_config[group_name]
    seq_att_map = feature_dict.seq_att_map
    keys = seq_att_map.key
    hist_seqs = seq_att_map.hist_seq
    tf_summary = feature_dict.tf_summary
    if tf_summary:
      logging.info('Write sequence feature to tensorflow summary.')

    def _seq_embed_summary_name(input_name):
      input_name = input_name.split(':')[0]
      input_name = input_name.split('/')[:2]
      return 'sequence_feature/' + '/'.join(input_name)

    with tf.variable_scope(group_name, reuse=tf.AUTO_REUSE):
      key_tensors = []
      for key in keys:
        qfc = feature_column_dict[key]
        with tf.variable_scope(qfc._var_scope_name):
          key_tensors.append(
              feature_column_dict[key]._get_dense_tensor(builder))
      if tf_summary:
        for key_tensor in key_tensors:
          tf.summary.histogram(
              _seq_embed_summary_name(key_tensor.name), key_tensor)

      hist_tensors = []
      for hist_seq in hist_seqs:
        seq_fc = feature_column_dict[hist_seq]
        with tf.variable_scope(seq_fc._var_scope_name):
          hist_tensors.append(
              feature_column_dict[hist_seq]._get_sequence_dense_tensor(builder))

      if tf_summary:
        for hist_embed, hist_seq_len in hist_tensors:
          tf.summary.histogram(
              _seq_embed_summary_name(hist_embed.name), hist_embed)
          tf.summary.histogram(
              _seq_embed_summary_name(hist_seq_len.name), hist_seq_len)

    features = {
        'key': tf.concat(key_tensors, axis=-1),
        'hist_seq_emb': tf.concat([x[0] for x in hist_tensors], axis=-1),
        'hist_seq_len': hist_tensors[0][1]
    }
    return features

  def get_wide_deep_dict(self):
    wide_and_deep_dict = {}
    for group_name_config in self._feature_groups_config.values():
      seq_att = group_name_config.seq_att_map
      for key in seq_att.key:
        wide_and_deep_dict[key] = WideOrDeep.DEEP
      for hist_seq in seq_att.hist_seq:
        wide_and_deep_dict[hist_seq] = WideOrDeep.DEEP
    return wide_and_deep_dict
