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
        self._feature_groups_config = {x.group_name: x for x in feature_groups_config}
        wide_and_deep_dict = self.get_wide_deep_dict()
        self._fc_parser = FeatureColumnParser(feature_configs, wide_and_deep_dict)

    def __call__(self, features, group_name, feature_name_to_output_tensors={}, allow_key_search=True):
        feature_column_dict = self._fc_parser.deep_columns
        feature_column_dict.update(self._fc_parser.sequence_columns)

        builder = feature_column._LazyBuilder(features)

        feature_dict = self._feature_groups_config[group_name]
        tf_summary = feature_dict.tf_summary
        if tf_summary:
            logging.info('Write sequence feature to tensorflow summary.')

        def _seq_embed_summary_name(input_name):
            input_name = input_name.split(':')[0]
            input_name = input_name.split('/')[:2]
            return 'sequence_feature/' + '/'.join(input_name)

        with tf.variable_scope(group_name, reuse=tf.AUTO_REUSE):
            key_tensors = []
            hist_tensors = []
            for x in feature_dict.seq_att_map:
                for key in x.key:
                    if key not in feature_name_to_output_tensors or (
                        feature_name_to_output_tensors[key] is None and allow_key_search
                    ):
                        qfc = feature_column_dict[key]
                        with tf.variable_scope(qfc._var_scope_name):
                            key_tensors.append(feature_column_dict[key]._get_dense_tensor(builder))
                    elif feature_name_to_output_tensors[key] is None:
                        assert feature_name_to_output_tensors[key] is not None, (
                            'When allow_key_search is False, key: %s should defined in same feature group.' % key
                        )
                    else:
                        key_tensors.append(feature_name_to_output_tensors[key])

                if tf_summary:
                    for key_tensor in key_tensors:
                        tf.summary.histogram(_seq_embed_summary_name(key_tensor.name), key_tensor)

                for hist_seq in x.hist_seq:
                    seq_fc = feature_column_dict[hist_seq]
                    with tf.variable_scope(seq_fc._var_scope_name):
                        hist_tensors.append(feature_column_dict[hist_seq]._get_sequence_dense_tensor(builder))

                if tf_summary:
                    for hist_embed, hist_seq_len in hist_tensors:
                        tf.summary.histogram(_seq_embed_summary_name(hist_embed.name), hist_embed)
                        tf.summary.histogram(_seq_embed_summary_name(hist_seq_len.name), hist_seq_len)

        features = {
            'key': tf.concat(key_tensors, axis=-1),
            'hist_seq_emb': tf.concat([x[0] for x in hist_tensors], axis=-1),
            'hist_seq_len': hist_tensors[0][1],
        }
        return features

    def get_wide_deep_dict(self):
        wide_and_deep_dict = {}
        for group_name_config in self._feature_groups_config.values():
            for x in group_name_config.seq_att_map:
                for key in x.key:
                    wide_and_deep_dict[key] = WideOrDeep.DEEP
                for hist_seq in x.hist_seq:
                    wide_and_deep_dict[hist_seq] = WideOrDeep.DEEP
        return wide_and_deep_dict
