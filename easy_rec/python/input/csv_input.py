# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.check_utils import check_split

if tf.__version__ >= '2.0':
  ignore_errors = tf.data.experimental.ignore_errors()
  tf = tf.compat.v1
else:
  ignore_errors = tf.contrib.data.ignore_errors()


class CSVInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(CSVInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    self._with_header = data_config.with_header
    self._field_names = None

  def _parse_csv(self, line):
    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    if self._field_names:
      # decode by csv header
      record_defaults = []
      for field_name in self._field_names:
        if field_name in self._input_fields:
          tid = self._input_fields.index(field_name)
          record_defaults.append(
              self.get_type_defaults(self._input_field_types[tid],
                                     self._input_field_defaults[tid]))
        else:
          record_defaults.append('')

    check_list = [
        tf.py_func(
            check_split, [
                line, self._data_config.separator,
                len(record_defaults), self._check_mode
            ],
            Tout=tf.bool)
    ] if self._check_mode else []
    with tf.control_dependencies(check_list):
      fields = tf.decode_csv(
          line,
          field_delim=self._data_config.separator,
          record_defaults=record_defaults,
          name='decode_csv')
      if self._field_names is not None:
        fields = [
            fields[self._field_names.index(x)] for x in self._input_fields
        ]

    # filter only valid fields
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}

    # filter only valid labels
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    file_paths = []
    for path in self._input_path:
      for x in tf.gfile.Glob(path):
        if not x.endswith('_SUCCESS'):
          file_paths.append(x)
    assert len(file_paths) > 0, 'match no files with %s' % self._input_path

    assert not file_paths[0].endswith(
        '.tar.gz'), 'could only support .csv or .gz(not .tar.gz) files.'

    compression_type = 'GZIP' if file_paths[0].endswith('.gz') else ''
    if compression_type:
      logging.info('compression_type = %s' % compression_type)

    if self._with_header:
      with tf.gfile.GFile(file_paths[0], 'r') as fin:
        for line_str in fin:
          line_str = line_str.strip()
          self._field_names = line_str.split(self._data_config.separator)
          break
        print('field_names: %s' % ','.join(self._field_names))

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
          lambda x: tf.data.TextLineDataset(
              x, compression_type=compression_type).skip(
                  int(self._with_header)),
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
    elif self._task_num > 1:  # For distribute evaluate
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      parallel_num = min(num_parallel_calls, len(file_paths))
      dataset = dataset.interleave(
          lambda x: tf.data.TextLineDataset(
              x, compression_type=compression_type).skip(
                  int(self._with_header)),
          cycle_length=parallel_num,
          num_parallel_calls=parallel_num)
      dataset = self._safe_shard(dataset)
      dataset = dataset.repeat(1)
    else:
      logging.info('eval files[%d]: %s' %
                   (len(file_paths), ','.join(file_paths)))
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      parallel_num = min(num_parallel_calls, len(file_paths))
      dataset = dataset.interleave(
          lambda x: tf.data.TextLineDataset(
              x, compression_type=compression_type).skip(
                  int(self._with_header)),
          cycle_length=parallel_num,
          num_parallel_calls=parallel_num)
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self._data_config.batch_size)
    dataset = dataset.map(
        self._parse_csv, num_parallel_calls=num_parallel_calls)
    if self._data_config.ignore_error:
      dataset = dataset.apply(ignore_errors)
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
