# -*- coding: utf-8 -*-
import logging
import os

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.hive_utils import HiveUtils


class HiveInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(HiveInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    if input_path is None:
      return
    self._data_config = data_config
    self._feature_config = feature_config
    self._hive_config = input_path

    hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    self._input_hdfs_path = hive_util.get_table_location(
        self._hive_config.table_name)
    self._input_table_col_names, self._input_table_col_types = hive_util.get_all_cols(
        self._hive_config.table_name)

  def _parse_csv(self, line):
    record_defaults = []
    for field_name in self._input_table_col_names:
      if field_name in self._input_fields:
        tid = self._input_fields.index(field_name)
        record_defaults.append(
            self.get_type_defaults(self._input_field_types[tid],
                                   self._input_field_defaults[tid]))
      else:
        record_defaults.append('')

    tmp_fields = tf.decode_csv(
        line,
        field_delim=self._data_config.separator,
        record_defaults=record_defaults,
        name='decode_csv')

    fields = []
    for x in self._input_fields:
      assert x in self._input_table_col_names, 'Column %s not in Table %s.' % (
          x, self._hive_config.table_name)
      fields.append(tmp_fields[self._input_table_col_names.index(x)])

    # filter only valid fields
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    file_paths = tf.gfile.Glob(os.path.join(self._input_hdfs_path, '*'))
    assert len(
        file_paths) > 0, 'match no files with %s' % self._hive_config.table_name

    num_parallel_calls = self._data_config.num_parallel_calls
    if mode == tf.estimator.ModeKeys.TRAIN:
      logging.info('train files[%d]: %s' %
                   (len(file_paths), ','.join(file_paths)))
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)

      if self._data_config.file_shard:
        dataset = self._safe_shard(dataset)

      if self._data_config.shuffle:
        # shuffle input files
        dataset = dataset.shuffle(len(file_paths))

      # too many readers read the same file will cause performance issues
      # as the same data will be read multiple times
      parallel_num = min(num_parallel_calls, len(file_paths))
      dataset = dataset.interleave(
          lambda x: tf.data.TextLineDataset(x),
          cycle_length=parallel_num,
          num_parallel_calls=parallel_num)

      if not self._data_config.file_shard:
        dataset = self._safe_shard(dataset)

      if self._data_config.shuffle:
        dataset = dataset.shuffle(
            self._data_config.shuffle_buffer_size,
            seed=2020,
            reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      logging.info('eval files[%d]: %s' %
                   (len(file_paths), ','.join(file_paths)))
      dataset = tf.data.TextLineDataset(file_paths)
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self._data_config.batch_size)
    dataset = dataset.map(
        self._parse_csv, num_parallel_calls=num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    dataset = dataset.map(
        map_func=self._preprocess, num_parallel_calls=num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
