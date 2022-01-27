# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf
import json
import rtp_fg
from tensorflow.python.framework import ops

from easy_rec.python.input.odps_rtp_input import OdpsRTPInput

try:
  import pai
except Exception:
  pass

class OdpsRTPInputV2(OdpsRTPInput):
  """RTPInput for parsing rtp fg new input format on odps.

  Our new format(csv in table) of rtp output:
     label0, item_id, ..., user_id, features
  Where features is in default RTP-tensorflow format.
  The features column and labels are specified by data_config.selected_cols,
     columns are selected by names in the table
     such as: clk,features, the last selected column is features, the first
     selected columns are labels
  """
  
  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1):
    super(OdpsRTPInputV2, self).__init__(
      data_config, feature_config, input_path, task_index, task_num)
    # load fg config
    if isinstance(self._input_path, tuple):
      if len(self._input_path) == 1:
        self._fg_config_path = None
      elif len(self._input_path) == 2:
        self._input_path, self._fg_config_path = self._input_path
      else:
        raise ValueError("illegal input path: {}".format(input_path))
    elif isinstance(self._input_path, str):
      self._input_path, self._fg_config_path = self._parse_input_path(self._input_path)
    else:
      self._fg_config_path = None
    logging.info('fg config path: {}'.format(self._fg_config_path))
    if self._fg_config_path is not None:
      with tf.gfile.GFile(self._fg_config_path, 'r') as f:
        self._fg_config = json.load(f)
    else:
      self._fg_config = None
  
  def _parse_input_path(input_path):
    if input_path is not None:
      parts = input_path.split(";")
      if len(parts) == 1:
        return (parts[0], None)
      elif len(parts) == 2:
        return (parts[0], parts[1])
      else:
        raise ValueError("illegal input path: {}".format(input_path))
    else:
      return (None, None)
  
  def _parse_table(self, *fields):
    fields = list(fields)
    labels = fields[:-1]

    # only for features, labels excluded
    record_defaults = [
        self.get_type_defaults(t, v)
        for x, t, v in zip(self._input_fields, self._input_field_types,
                           self._input_field_defaults)
        if x not in self._label_fields
    ]
    # assume that the last field is the generated feature column
    features = rtp_fg.parse_genreated_fg(self._fg_config, fields[-1])

    field_keys = [x for x in self._input_fields if x not in self._label_fields]
    for feature_key in features:
      if feature_key not in field_keys or feature_key not in self._effective_fields:
        del features[feature_key]
    inputs = {x: features[x] for x in features.keys()}

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]
    return inputs
