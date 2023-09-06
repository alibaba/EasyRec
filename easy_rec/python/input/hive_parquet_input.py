# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.hive_utils import HiveUtils
from easy_rec.python.utils.tf_utils import get_tf_type


class HiveParquetInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(HiveParquetInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    if input_path is None:
      return
    self._data_config = data_config
    self._feature_config = feature_config
    self._hive_config = input_path

    hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    input_hdfs_path = hive_util.get_table_location(self._hive_config.table_name)
    self._input_table_col_names, self._input_table_col_types = hive_util.get_all_cols(
        self._hive_config.table_name)
    self._all_hdfs_path = tf.gfile.Glob(os.path.join(input_hdfs_path, '*'))

    for x in self._input_fields:
      assert x in self._input_table_col_names, 'Column %s not in Table %s.' % (
          x, self._hive_config.table_name)

    self._record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

  def _file_shard(self, file_paths, task_num, task_index):
    if self._data_config.chief_redundant:
      task_num = max(task_num - 1, 1)
      task_index = max(task_index - 1, 0)
    task_file_paths = []
    for idx in range(task_index, len(file_paths), task_num):
      task_file_paths.append(file_paths[idx])
    return task_file_paths

  def _parquet_read(self):
    for input_path in self._input_hdfs_path:
      if input_path.endswith('SUCCESS'):
        continue
      df = pd.read_parquet(input_path, engine='pyarrow')
      df = df[self._input_fields]
      df.replace('', np.nan, inplace=True)
      df.replace('NULL', np.nan, inplace=True)
      total_records_num = len(df)

      for k, v in zip(self._input_fields, self._record_defaults):
        df[k].fillna(v, inplace=True)

      for start_idx in range(0, total_records_num,
                             self._data_config.batch_size):
        end_idx = min(total_records_num,
                      start_idx + self._data_config.batch_size)
        batch_data = df[start_idx:end_idx]
        inputs = []
        for k in self._input_fields:
          inputs.append(batch_data[k].to_numpy())
        yield tuple(inputs)

  def _parse_csv(self, *fields):
    # filter only valid fields
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    # filter only valid labels
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    # get input type
    list_type = [get_tf_type(x) for x in self._input_field_types]
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    if len(self._all_hdfs_path) >= 2 * self._task_num:
      file_shard = True
      self._input_hdfs_path = self._file_shard(self._all_hdfs_path,
                                               self._task_num, self._task_index)
    else:
      file_shard = False
      self._input_hdfs_path = self._all_hdfs_path
    logging.info('input path: %s' % self._input_hdfs_path)
    assert len(self._input_hdfs_path
               ) > 0, 'match no files with %s' % self._hive_config.table_name

    dataset = tf.data.Dataset.from_generator(
        self._parquet_read, output_types=list_type, output_shapes=list_shapes)

    if not file_shard:
      dataset = self._safe_shard(dataset)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(
          self._data_config.shuffle_buffer_size,
          seed=2020,
          reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.map(
        self._parse_csv,
        num_parallel_calls=self._data_config.num_parallel_calls)

    # preprocess is necessary to transform data
    # so that they could be feed into FeatureColumns
    dataset = dataset.map(
        map_func=self._preprocess,
        num_parallel_calls=self._data_config.num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
