# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.tf_utils import get_tf_type

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class TFRecordInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(TFRecordInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

    self.feature_desc = {}
    for x, t, d in zip(self._input_fields, self._input_field_types,
                       self._input_field_defaults):
      d = self.get_type_defaults(t, d)
      t = get_tf_type(t)
      self.feature_desc[x] = tf.FixedLenFeature(
          dtype=t, shape=1, default_value=d)

  def _parse_tfrecord(self, example):
    try:
      inputs = tf.parse_single_example(example, features=self.feature_desc)
    except AttributeError:
      inputs = tf.io.parse_single_example(example, features=self.feature_desc)
    return inputs

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    file_paths = []
    for x in self._input_path:
      file_paths.extend(tf.gfile.Glob(x))
    assert len(file_paths) > 0, 'match no files with %s' % self._input_path

    num_parallel_calls = self._data_config.num_parallel_calls
    data_compression_type = self._data_config.data_compression_type
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
          lambda x: tf.data.TFRecordDataset(
              x, compression_type=data_compression_type),
          cycle_length=parallel_num,
          num_parallel_calls=parallel_num)
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
      dataset = tf.data.TFRecordDataset(
          file_paths, compression_type=data_compression_type)
      dataset = dataset.repeat(1)

    dataset = dataset.map(
        self._parse_tfrecord, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(self._data_config.batch_size)
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
