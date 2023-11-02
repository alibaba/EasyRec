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
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils.check_utils import check_split

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CSVPredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               ds_vector_recall=False,
               fg_json_path=None,
               profiling_file=None,
               selected_cols=None,
               output_sep=chr(1)):
    super(CSVPredictor, self).__init__(model_path, profiling_file, fg_json_path)
    self._output_sep = output_sep
    self._ds_vector_recall = ds_vector_recall
    input_type = DatasetConfig.InputType.Name(data_config.input_type).lower()
    self._with_header = data_config.with_header

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
    check_list = [
        tf.py_func(
            check_split, [line, self._input_sep,
                          len(self._record_defaults)],
            Tout=tf.bool)
    ]
    with tf.control_dependencies(check_list):
      fields = tf.decode_csv(
          line,
          field_delim=self._input_sep,
          record_defaults=self._record_defaults,
          name='decode_csv')
    if self._is_rtp:
      if self._with_header:
        inputs = dict(zip(self._all_fields, fields))
      else:
        inputs = {}
        idx = 0
        for x in range(len(self._record_defaults) - 1):
          if not self._selected_cols or x in self._selected_cols[:-1]:
            inputs[self._input_fields[idx]] = fields[x]
            idx += 1
          else:
            inputs['no_used_%d' % x] = fields[x]
        inputs[SINGLE_PLACEHOLDER_FEATURE_KEY] = fields[-1]
    else:
      inputs = {self._all_fields[x]: fields[x] for x in range(len(fields))}
    return inputs

  def _get_num_cols(self, file_paths):
    # try to figure out number of fields from one file
    num_cols = -1
    with gfile.GFile(file_paths[0], 'r') as fin:
      num_lines = 0
      for line_str in fin:
        line_tok = line_str.strip().split(self._input_sep)
        if num_cols != -1:
          assert num_cols == len(line_tok), (
              'num selected cols is %d, not equal to %d, current line is: %s, please check input_sep and data.'
              % (num_cols, len(line_tok), line_str))
        num_cols = len(line_tok)
        num_lines += 1
        if num_lines > 10:
          break
    logging.info('num selected cols = %d' % num_cols)
    return num_cols

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    file_paths = []
    for path in input_path.split(','):
      for x in gfile.Glob(path):
        if not x.endswith('_SUCCESS'):
          file_paths.append(x)
    assert len(file_paths) > 0, 'match no files with %s' % input_path

    if self._with_header:
      with gfile.GFile(file_paths[0], 'r') as fin:
        for line_str in fin:
          line_str = line_str.strip()
          self._field_names = line_str.split(self._input_sep)
          break
      print('field_names: %s' % ','.join(self._field_names))
      self._all_fields = self._field_names
    elif self._ds_vector_recall:
      self._all_fields = self._selected_cols
    else:
      self._all_fields = self._input_fields
    if self._is_rtp:
      num_cols = self._get_num_cols(file_paths)
      self._record_defaults = ['' for _ in range(num_cols)]
      if not self._selected_cols:
        self._selected_cols = list(range(num_cols))
      for col_idx in self._selected_cols[:-1]:
        col_name = self._input_fields[col_idx]
        default_val = self._get_defaults(col_name)
        self._record_defaults[col_idx] = default_val
    else:
      self._record_defaults = [
          self._get_defaults(col_name) for col_name in self._all_fields
      ]

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    parallel_num = min(num_parallel_calls, len(file_paths))
    dataset = dataset.interleave(
        lambda x: tf.data.TextLineDataset(x).skip(int(self._with_header)),
        cycle_length=parallel_num,
        num_parallel_calls=parallel_num)
    dataset = dataset.shard(slice_num, slice_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

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
