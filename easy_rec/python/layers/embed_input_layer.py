# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.feature_column.feature_group import FeatureGroup


class EmbedInputLayer(object):

  def __init__(self, feature_groups_config, dump_dir=None):
    self._feature_groups = {
        x.group_name: FeatureGroup(x) for x in feature_groups_config
    }
    self._dump_dir = dump_dir

  def __call__(self, features, group_name):
    assert group_name in self._feature_groups, 'invalid group_name[%s], list: %s' % \
                                               ','.join([x for x in self._feature_groups])
    feature_group = self._feature_groups[group_name]
    group_features = []
    for feature_name in feature_group.feature_names:
      tmp_fea = features[feature_name]
      group_features.append(tmp_fea)
    return tf.concat(group_features, axis=1), group_features
