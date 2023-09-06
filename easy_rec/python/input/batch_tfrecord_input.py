# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.tf_utils import get_tf_type

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class BatchTFRecordInput(Input):
  """BatchTFRecordInput use for batch read from tfrecord.

  For example, there is a tfrecord which one feature(key)
  correspond to n data(value).
  batch_size needs to be a multiple of n.
  """

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(BatchTFRecordInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    assert data_config.HasField(
        'n_data_batch_tfrecord'), 'Need to set n_data_batch_tfrecord in config.'
    self._input_shapes = [x.input_shape for x in data_config.input_fields]
    self.feature_desc = {}
    for x, t, d, s in zip(self._input_fields, self._input_field_types,
                          self._input_field_defaults, self._input_shapes):
      d = self.get_type_defaults(t, d)
      t = get_tf_type(t)
      self.feature_desc[x] = tf.io.FixedLenSequenceFeature(
          dtype=t, shape=s, allow_missing=True)

  def _parse_tfrecord(self, example):
    try:
      _, features, _ = tf.parse_sequence_example(
          example, sequence_features=self.feature_desc)
    except AttributeError:
      _, features, _ = tf.io.parse_sequence_example(
          example, sequence_features=self.feature_desc)
    # Below code will reduce one dimension when the data dimension > 2.
    features = dict(
        (key,
         tf.reshape(value, [
             -1,
         ] + [x for i, x in enumerate(value.shape) if i not in (0, 1)])) for (
             key, value) in features.items())
    return features

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

    # We read n data from tfrecord one time.
    cur_batch = self._data_config.batch_size // self._data_config.n_data_batch_tfrecord
    cur_batch = max(1, cur_batch)
    dataset = dataset.batch(cur_batch)
    dataset = dataset.map(
        self._parse_tfrecord, num_parallel_calls=num_parallel_calls)

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
