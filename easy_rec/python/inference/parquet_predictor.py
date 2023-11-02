# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.inference.predictor import SINGLE_PLACEHOLDER_FEATURE_KEY
from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.input.parquet_input import ParquetInput
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils import input_utils

try:
  from tensorflow.python.framework.load_library import load_op_library
  import easy_rec
  load_embed_lib_path = os.path.join(easy_rec.ops_dir, 'libload_embed.so')
  load_embed_lib = load_op_library(load_embed_lib_path)
except Exception as ex:
  logging.warning('load libload_embed.so failed: %s' % str(ex))


class ParquetPredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               ds_vector_recall=False,
               fg_json_path=None,
               profiling_file=None,
               selected_cols=None,
               output_sep=chr(1),
               pipeline_config=None):
    super(ParquetPredictor, self).__init__(model_path, profiling_file,
                                           fg_json_path)
    self._output_sep = output_sep
    self._ds_vector_recall = ds_vector_recall
    input_type = DatasetConfig.InputType.Name(data_config.input_type).lower()
    self.pipeline_config = pipeline_config

    if 'rtp' in input_type:
      self._is_rtp = True
      self._input_sep = data_config.rtp_separator
    else:
      self._is_rtp = False
      self._input_sep = data_config.separator

    if selected_cols and not ds_vector_recall:
      self._selected_cols = [int(x) for x in selected_cols.split(',')]
    elif ds_vector_recall:
      self._selected_cols = selected_cols.split(',')
    else:
      self._selected_cols = None

  def _get_reserved_cols(self, reserved_cols):
    if reserved_cols == 'ALL_COLUMNS':
      if self._is_rtp:
        if self._with_header:
          reserved_cols = self._all_fields
        else:
          idx = 0
          reserved_cols = []
          for x in range(len(self._record_defaults) - 1):
            if not self._selected_cols or x in self._selected_cols[:-1]:
              reserved_cols.append(self._input_fields[idx])
              idx += 1
            else:
              reserved_cols.append('no_used_%d' % x)
          reserved_cols.append(SINGLE_PLACEHOLDER_FEATURE_KEY)
      else:
        reserved_cols = self._all_fields
    else:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, line):
    out_dict = {}
    for key in line['feature']:
      out_dict[key] = line['feature'][key]
    if 'reserve' in line:
      for key in line['reserve']:
        out_dict[key] = line['reserve'][key]
    return out_dict

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    feature_configs = config_util.get_compatible_feature_configs(
        self.pipeline_config)
    kwargs = {}
    if self.pipeline_config.model_config.HasField('ev_params'):
      kwargs['ev_params'] = self.pipeline_config.model_config.ev_params
    if self._reserved_cols is not None and len(self._reserved_cols) > 0:
      kwargs['reserve_fields'] = self._reserved_cols
      parquet_file = gfile.Glob(input_path.split(',')[0])[0]
      kwargs['reserve_types'] = input_utils.get_tf_type_from_parquet_file(
          self._reserved_cols, parquet_file)
    parquet_input = ParquetInput(
        self.pipeline_config.data_config,
        feature_configs,
        input_path,
        task_index=slice_id,
        task_num=slice_num,
        pipeline_config=self.pipeline_config,
        **kwargs)
    return parquet_input._build(tf.estimator.ModeKeys.PREDICT, {})

  def _get_writer(self, output_path, slice_id):
    if not gfile.Exists(output_path):
      gfile.MakeDirs(output_path)
    res_path = os.path.join(output_path, 'part-%d.csv' % slice_id)
    table_writer = gfile.GFile(res_path, 'w')
    table_writer.write(
        self._output_sep.join(self._output_cols + self._reserved_cols) + '\n')
    return table_writer

  def _write_lines(self, table_writer, outputs):
    outputs = '\n'.join(
        [self._output_sep.join([str(i) for i in output]) for output in outputs])
    table_writer.write(outputs + '\n')

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [outputs[x] for x in output_cols] + \
                   [all_vals[k] for k in reserved_cols]
    return reserve_vals

  @property
  def out_of_range_exception(self):
    return (tf.errors.OutOfRangeError)
