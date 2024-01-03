# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

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

  def _parse_line(self, line):
    out_dict = {}
    for key in line['feature']:
      out_dict[key] = line['feature'][key]
    if 'reserve' in line:
      out_dict['reserve'] = line['reserve']
    #   for key in line['reserve']:
    #     out_dict[key] = line['reserve'][key]
    return out_dict

  def _get_reserved_cols(self, reserved_cols):
    # already parsed in _get_dataset
    return self._reserved_cols

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    feature_configs = config_util.get_compatible_feature_configs(
        self.pipeline_config)

    kwargs = {}
    if self._reserved_args is not None and len(self._reserved_args) > 0:
      if self._reserved_args == 'ALL_COLUMNS':
        parquet_file = gfile.Glob(input_path.split(',')[0])[0]
        # gfile not supported, read_parquet requires random access
        all_data = pd.read_parquet(parquet_file)
        all_cols = list(all_data.columns)
        kwargs['reserve_fields'] = all_cols
        self._all_fields = all_cols
        self._reserved_cols = all_cols
        kwargs['reserve_types'] = input_utils.get_tf_type_from_parquet_file(
            all_cols, parquet_file)
      else:
        self._reserved_cols = [
            x.strip() for x in self._reserved_args.split(',') if x.strip() != ''
        ]
        kwargs['reserve_fields'] = self._reserved_cols
        parquet_file = gfile.Glob(input_path.split(',')[0])[0]
        kwargs['reserve_types'] = input_utils.get_tf_type_from_parquet_file(
            self._reserved_cols, parquet_file)
      logging.info('reserve_fields=%s reserve_types=%s' %
                   (','.join(self._reserved_cols), ','.join(
                       [str(x) for x in kwargs['reserve_types']])))
    else:
      self._reserved_cols = []
    self.pipeline_config.data_config.batch_size = batch_size

    kwargs['is_predictor'] = True
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
    reserve_vals = []
    for x in outputs:
      tmp_val = outputs[x]
      reserve_vals.append(tmp_val)
    for k in reserved_cols:
      tmp_val = all_vals['reserve'][k]
      if tmp_val.dtype == np.object:
        tmp_val = [x.decode('utf-8') for x in tmp_val]
      reserve_vals.append(tmp_val)
    return reserve_vals

  @property
  def out_of_range_exception(self):
    return (tf.errors.OutOfRangeError)
