# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import re

from easy_rec.python.protos.feature_config_pb2 import FeatureGroupConfig
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep


class FeatureGroup(object):

  def __init__(self, feature_group_config):
    self._config = feature_group_config
    assert isinstance(self._config, FeatureGroupConfig)
    assert self._config.wide_deep in [WideOrDeep.WIDE, WideOrDeep.DEEP]
    self._auto_expand_feature_name()

  @property
  def group_name(self):
    return self._config.group_name

  @property
  def wide_and_deep_dict(self):
    wide_and_deep_dict = {}
    for feature_name in self._config.feature_names:
      wide_and_deep_dict[feature_name] = self._config.wide_deep
    return wide_and_deep_dict

  @property
  def feature_names(self):
    return self._config.feature_names

  def select_columns(self, fc):
    columns = fc.wide_columns if self._config.wide_deep == WideOrDeep.WIDE \
        else fc.deep_columns
    return [columns[x] for x in self._config.feature_names]

  def _auto_expand_feature_name(self):
    features = [x for x in self._config.feature_names]
    while len(self._config.feature_names) > 0:
      self._config.feature_names.pop()
    for feature in features:
      match_obj = re.match(r'([a-zA-Z_]+)\[([0-9]+)-([0-9]+)\]', feature)
      if match_obj:
        prefix = match_obj.group(1)
        sid = int(match_obj.group(2))
        eid = int(match_obj.group(3)) + 1
        for tid in range(sid, eid):
          tmp_f = '%s%d' % (prefix, tid)
          self._config.feature_names.append(tmp_f)
      else:
        self._config.feature_names.append(feature)
