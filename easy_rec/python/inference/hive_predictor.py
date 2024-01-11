# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import tf_utils
from easy_rec.python.utils.hive_utils import HiveUtils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class HivePredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               hive_config,
               fg_json_path=None,
               profiling_file=None,
               output_sep=chr(1),
               all_cols=None,
               all_col_types=None):
    super(HivePredictor, self).__init__(model_path, profiling_file,
                                        fg_json_path)

    self._data_config = data_config
    self._hive_config = hive_config
    self._output_sep = output_sep
    input_type = DatasetConfig.InputType.Name(data_config.input_type).lower()
    if 'rtp' in input_type:
      self._is_rtp = True
    else:
      self._is_rtp = False
    self._all_cols = [x.strip() for x in all_cols if x != '']
    self._all_col_types = [x.strip() for x in all_col_types if x != '']
    self._record_defaults = [
        self._get_defaults(col_name, col_type)
        for col_name, col_type in zip(self._all_cols, self._all_col_types)
    ]

  def _get_reserved_cols(self, reserved_cols):
    if reserved_cols == 'ALL_COLUMNS':
      reserved_cols = self._all_cols
    else:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, line):
    field_delim = self._data_config.rtp_separator if self._is_rtp else self._data_config.separator
    fields = tf.decode_csv(
        line,
        field_delim=field_delim,
        record_defaults=self._record_defaults,
        name='decode_csv')
    inputs = {self._all_cols[x]: fields[x] for x in range(len(fields))}
    return inputs

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    self._hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    self._input_hdfs_path = self._hive_util.get_table_location(input_path)
    file_paths = tf.gfile.Glob(os.path.join(self._input_hdfs_path, '*'))
    assert len(file_paths) > 0, 'match no files with %s' % input_path

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    parallel_num = min(num_parallel_calls, len(file_paths))
    dataset = dataset.interleave(
        tf.data.TextLineDataset,
        cycle_length=parallel_num,
        num_parallel_calls=parallel_num)
    dataset = dataset.shard(slice_num, slice_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  def get_table_info(self, output_path):
    partition_name, partition_val = None, None
    if len(output_path.split('/')) == 2:
      table_name, partition = output_path.split('/')
      partition_name, partition_val = partition.split('=')
    else:
      table_name = output_path
    return table_name, partition_name, partition_val

  def _get_writer(self, output_path, slice_id):
    table_name, partition_name, partition_val = self.get_table_info(output_path)
    is_exist = self._hive_util.is_table_or_partition_exist(
        table_name, partition_name, partition_val)
    assert not is_exist, '%s is already exists. Please drop it.' % output_path

    output_path = output_path.replace('.', '/')
    self._hdfs_path = 'hdfs://%s:9000/user/easy_rec/%s_tmp' % (
        self._hive_config.host, output_path)
    if not gfile.Exists(self._hdfs_path):
      gfile.MakeDirs(self._hdfs_path)
    res_path = os.path.join(self._hdfs_path, 'part-%d.csv' % slice_id)
    table_writer = gfile.GFile(res_path, 'w')
    return table_writer

  def _write_lines(self, table_writer, outputs):
    outputs = '\n'.join(
        [self._output_sep.join([str(i) for i in output]) for output in outputs])
    table_writer.write(outputs + '\n')

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [outputs[x] for x in output_cols] + \
                   [all_vals[k] for k in reserved_cols]
    return reserve_vals

  def load_to_table(self, output_path, slice_num, slice_id):
    res_path = os.path.join(self._hdfs_path, 'SUCCESS-%s' % slice_id)
    success_writer = gfile.GFile(res_path, 'w')
    success_writer.write('')
    success_writer.close()

    if slice_id != 0:
      return

    for id in range(slice_num):
      res_path = os.path.join(self._hdfs_path, 'SUCCESS-%s' % id)
      while not gfile.Exists(res_path):
        time.sleep(10)

    table_name, partition_name, partition_val = self.get_table_info(output_path)
    schema = ''
    for output_col_name in self._output_cols:
      tf_type = self._predictor_impl._outputs_map[output_col_name].dtype
      col_type = tf_utils.get_col_type(tf_type)
      schema += output_col_name + ' ' + col_type + ','

    for output_col_name in self._reserved_cols:
      assert output_col_name in self._all_cols, 'Column: %s not exists.' % output_col_name
      idx = self._all_cols.index(output_col_name)
      output_col_types = self._all_col_types[idx]
      schema += output_col_name + ' ' + output_col_types + ','
    schema = schema.rstrip(',')

    if partition_name and partition_val:
      sql = 'create table if not exists %s (%s) PARTITIONED BY (%s string)' % \
            (table_name, schema, partition_name)
      self._hive_util.run_sql(sql)
      sql = "LOAD DATA INPATH '%s/*' INTO TABLE %s PARTITION (%s=%s)" % \
            (self._hdfs_path, table_name, partition_name, partition_val)
      self._hive_util.run_sql(sql)
    else:
      sql = 'create table if not exists %s (%s)' % \
            (table_name, schema)
      self._hive_util.run_sql(sql)
      sql = "LOAD DATA INPATH '%s/*' INTO TABLE %s" % \
            (self._hdfs_path, table_name)
      self._hive_util.run_sql(sql)

  @property
  def out_of_range_exception(self):
    return (tf.errors.OutOfRangeError)
