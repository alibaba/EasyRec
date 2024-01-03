# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import tf_utils
from easy_rec.python.utils.hive_utils import HiveUtils
from easy_rec.python.utils.tf_utils import get_tf_type

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class HiveParquetPredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               hive_config,
               fg_json_path=None,
               profiling_file=None,
               output_sep=chr(1),
               all_cols=None,
               all_col_types=None):
    super(HiveParquetPredictor, self).__init__(model_path, profiling_file,
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

  def _parse_line(self, *fields):
    fields = list(fields)
    field_dict = {self._all_cols[i]: fields[i] for i in range(len(fields))}
    return field_dict

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    self._hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    hdfs_path = self._hive_util.get_table_location(input_path)
    self._input_hdfs_path = gfile.Glob(os.path.join(hdfs_path, '*'))
    assert len(self._input_hdfs_path) > 0, 'match no files with %s' % input_path

    list_type = []
    input_field_type_map = {
        x.input_name: x.input_type for x in self._data_config.input_fields
    }
    type_2_tftype = {
        'string': tf.string,
        'double': tf.double,
        'float': tf.float32,
        'bigint': tf.int32,
        'boolean': tf.bool
    }
    for col_name, col_type in zip(self._all_cols, self._all_col_types):
      if col_name in input_field_type_map:
        list_type.append(get_tf_type(input_field_type_map[col_name]))
      else:
        list_type.append(type_2_tftype[col_type.lower()])
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    def parquet_read():
      for input_path in self._input_hdfs_path:
        if input_path.endswith('SUCCESS'):
          continue
        df = pd.read_parquet(input_path, engine='pyarrow')

        df.replace('', np.nan, inplace=True)
        df.replace('NULL', np.nan, inplace=True)
        total_records_num = len(df)

        for k, v in zip(self._all_cols, self._record_defaults):
          df[k].fillna(v, inplace=True)

        for start_idx in range(0, total_records_num, batch_size):
          end_idx = min(total_records_num, start_idx + batch_size)
          batch_data = df[start_idx:end_idx]
          inputs = []
          for k in self._all_cols:
            inputs.append(batch_data[k].to_numpy())
          yield tuple(inputs)

    dataset = tf.data.Dataset.from_generator(
        parquet_read, output_types=list_type, output_shapes=list_shapes)
    dataset = dataset.shard(slice_num, slice_id)
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
