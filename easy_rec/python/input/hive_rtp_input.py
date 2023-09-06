# -*- coding: utf-8 -*-
import logging
import os

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.check_utils import check_split
from easy_rec.python.utils.hive_utils import HiveUtils
from easy_rec.python.utils.input_utils import string_to_number


class HiveRTPInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(HiveRTPInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    if input_path is None:
      return
    self._data_config = data_config
    self._feature_config = feature_config
    self._hive_config = input_path

    logging.info('input_fields: %s label_fields: %s' %
                 (','.join(self._input_fields), ','.join(self._label_fields)))

    self._rtp_separator = self._data_config.rtp_separator
    if not isinstance(self._rtp_separator, str):
      self._rtp_separator = self._rtp_separator.encode('utf-8')
    logging.info('rtp separator = %s' % self._rtp_separator)
    self._selected_cols = [c.strip() for c in self._data_config.selected_cols.split(',')] \
        if self._data_config.selected_cols else None
    logging.info('select cols: %s' % self._selected_cols)
    hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    self._input_hdfs_path = hive_util.get_table_location(
        self._hive_config.table_name)
    self._input_table_col_names, self._input_table_col_types = hive_util.get_all_cols(
        self._hive_config.table_name)

  def _parse_csv(self, line):
    non_feature_cols = self._label_fields
    if self._selected_cols:
      non_feature_cols = self._selected_cols[:-1]
    record_defaults = []
    for tid, field_name in enumerate(self._input_table_col_names):
      if field_name in self._selected_cols[:-1]:
        idx = self._input_fields.index(field_name)
        record_defaults.append(
            self.get_type_defaults(self._input_field_types[idx],
                                   self._input_field_defaults[idx]))
      else:
        record_defaults.append('')
    print('record_defaults: ', record_defaults)
    tmp_fields = tf.decode_csv(
        line,
        field_delim=self._rtp_separator,
        record_defaults=record_defaults,
        name='decode_csv')
    print('tmp_fields: ', tmp_fields)

    fields = []
    if self._selected_cols:
      for idx, field_name in enumerate(self._input_table_col_names):
        if field_name in self._selected_cols:
          fields.append(tmp_fields[idx])
    print('fields: ', fields)
    labels = fields[:-1]

    # only for features, labels and sample_weight excluded
    record_types = [
        t for x, t in zip(self._input_fields, self._input_field_types)
        if x not in non_feature_cols
    ]
    feature_num = len(record_types)

    check_list = [
        tf.py_func(
            check_split,
            [fields[-1], self._data_config.separator,
             len(record_types)],
            Tout=tf.bool)
    ] if self._check_mode else []
    with tf.control_dependencies(check_list):
      fields = tf.string_split(
          fields[-1], self._data_config.separator, skip_empty=False)
    tmp_fields = tf.reshape(fields.values, [-1, feature_num])

    rtp_record_defaults = [
        str(self.get_type_defaults(t, v))
        for x, t, v in zip(self._input_fields, self._input_field_types,
                           self._input_field_defaults)
        if x not in non_feature_cols
    ]
    fields = labels[len(self._label_fields):]
    for i in range(feature_num):
      field = string_to_number(tmp_fields[:, i], record_types[i],
                               rtp_record_defaults[i], i)
      fields.append(field)

    field_keys = [x for x in self._input_fields if x not in self._label_fields]
    effective_fids = [field_keys.index(x) for x in self._effective_fields]
    inputs = {field_keys[x]: fields[x] for x in effective_fids}

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]
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
