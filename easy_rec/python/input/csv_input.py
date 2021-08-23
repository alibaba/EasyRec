# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input

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
               task_num=1):
    super(CSVInput, self).__init__(data_config, feature_config, input_path,
                                   task_index, task_num)

  def _parse_csv(self, line):
    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    def _check_data(line):
      sep = self._data_config.separator
      if type(sep) != type(str):
        sep = sep.encode('utf-8')
      field_num = len(line[0].split(sep))
      assert field_num == len(record_defaults), \
          'sep[%s] maybe invalid: field_num=%d, required_num=%d' % \
          (sep, field_num, len(record_defaults))
      return True

    check_op = tf.py_func(_check_data, [line], Tout=tf.bool)
    with tf.control_dependencies([check_op]):
      fields = tf.decode_csv(
          line,
          field_delim=self._data_config.separator,
          record_defaults=record_defaults,
          name='decode_csv')

    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}

    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    file_paths = []
    for x in self._input_path.split(','):
      file_paths.extend(tf.gfile.Glob(x))
    assert len(file_paths) > 0, 'match no files with %s' % self._input_path

    num_parallel_calls = self._data_config.num_parallel_calls
    if mode == tf.estimator.ModeKeys.TRAIN:
      logging.info('train files[%d]: %s' %
                   (len(file_paths), ','.join(file_paths)))
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      if self._data_config.shuffle:
        # shuffle input files
        dataset = dataset.shuffle(len(file_paths))
      # too many readers read the same file will cause performance issues
      # as the same data will be read multiple times
      parallel_num = min(num_parallel_calls, len(file_paths))
      dataset = dataset.interleave(
          tf.data.TextLineDataset,
          cycle_length=parallel_num,
          num_parallel_calls=parallel_num)

      if self._data_config.chief_redundant:
        dataset = dataset.shard(
            max(self._task_num - 1, 1), max(self._task_index - 1, 0))
      else:
        dataset = dataset.shard(self._task_num, self._task_index)
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
