# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

import tensorflow as tf

import easy_rec
from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.feature_column.feature_group import FeatureGroup
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep

from easy_rec.python.compat.feature_column.feature_column import _SharedEmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import EmbeddingColumn  # NOQA

if tf.__version__ >= '2.0':
  gfile = tf.compat.v1.gfile
else:
  gfile = tf.gfile


class InputLayer(object):
  """Input Layer for generate input features.

  This class apply feature_columns to input tensors to generate wide features and deep features.
  """

  def __init__(self,
               feature_configs,
               feature_groups_config,
               wide_output_dim=-1,
               use_embedding_variable=False):
    self._feature_groups = {
        x.group_name: FeatureGroup(x) for x in feature_groups_config
    }
    wide_and_deep_dict = self.get_wide_deep_dict()
    self._fc_parser = FeatureColumnParser(
        feature_configs,
        wide_and_deep_dict,
        wide_output_dim,
        use_embedding_variable=use_embedding_variable)

  def has_group(self, group_name):
    return group_name in self._feature_groups

  def __call__(self, features, group_name):
    assert group_name in self._feature_groups, 'invalid group_name[%s], list: %s' % \
                                               (group_name, ','.join([x for x in self._feature_groups]))
    feature_group = self._feature_groups[group_name]
    group_columns = feature_group.select_columns(self._fc_parser)
    cols_to_output_tensors = {}
    features = feature_column.input_layer(
        features, group_columns, cols_to_output_tensors=cols_to_output_tensors)

    # dump model inputs
    self._dump_feature_shape(group_name, group_columns, cols_to_output_tensors)

    group_features = [cols_to_output_tensors[x] for x in group_columns]
    return features, group_features

  def _dump_feature_shape(self, group_name, group_columns,
                          cols_to_output_tensors):
    """Dump embedding feature column shape info.

    For large embedding serving on eas, the shapes are dumped so that
    embedding placeholders could be create in easy_rec/python/input/input.py

    Args:
      group_name: feature group name
      group_columns: feature columns of the group
      cols_to_output_tensors: dict of feature_columns to feature tensors
    """
    if 'dump_embedding_shape_dir' not in easy_rec._global_config:
      return
    dump_dir = easy_rec._global_config['dump_embedding_shape_dir']
    dump_path = os.path.join(dump_dir, 'input_layer_%s.txt' % group_name)
    with gfile.GFile(dump_path, 'w') as fout:
      for key in group_columns:
        assert 'name' in dir(key), 'column(%s) has no attributes name: %s' % \
                                   (str(key), str(dir(key)))
        if tf.__version__ >= '2.0':
          shape_vals = [x for x in cols_to_output_tensors[key].shape]
        else:
          shape_vals = [x.value for x in cols_to_output_tensors[key].shape]
        shape_config = {'name': key.name, 'shape': shape_vals}
        if isinstance(key, _SharedEmbeddingColumn):
          shape_config['embedding_name'] = key.shared_embedding_collection_name
        elif isinstance(key, EmbeddingColumn):
          shape_config['embedding_name'] = key.name.replace('_embedding', '')
        fout.write('%s\n' % json.dumps(shape_config))
    logging.info('dump input_layer to %s' % dump_path)

  def get_wide_deep_dict(self):
    """Get wide or deep indicator for feature columns.

    Returns:
      dict of { feature_name : WideOrDeep }
    """
    wide_and_deep_dict = {}
    for fg_name in self._feature_groups.keys():
      fg = self._feature_groups[fg_name]
      tmp_dict = fg.wide_and_deep_dict
      for k in tmp_dict:
        v = tmp_dict[k]
        if k not in wide_and_deep_dict:
          wide_and_deep_dict[k] = v
        elif wide_and_deep_dict[k] != v:
          wide_and_deep_dict[k] = WideOrDeep.WIDE_AND_DEEP
        else:
          pass
    return wide_and_deep_dict
