# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.protos.dataset_pb2 import DatasetConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class RTPInputV2(Input):
  """RTPInput for parsing rtp fg input format.

  the original rtp format, it is not efficient for training, the performance have to be tuned.
  """

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(RTPInputV2,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

  def _parse_rtp(self, lines):
    tf_types = [tf.string for x in self._input_field_types]

    def _parse_one_line_tf(line):
      line = tf.expand_dims(line, axis=0)
      field_toks = tf.string_split(line, '\002').values
      field_vals = tf.string_split(field_toks, '\003').values
      field_vals = tf.reshape(field_vals, [-1, 2])
      keys = field_vals[:, 0]
      vals = field_vals[:, 1]
      temp_vals = [
          str(
              self.get_type_defaults(self._input_field_types[i],
                                     self._input_field_defaults[i]))
          for i in range(len(self._input_fields))
      ]
      for i, key in enumerate(self._input_fields):
        msk = tf.equal(key, keys)
        val = tf.boolean_mask(vals, msk)
        def_val = self.get_type_defaults(self._input_field_types[i],
                                         self._input_field_defaults[i])
        temp_vals[i] = tf.cond(
            tf.reduce_any(msk), lambda: tf.reduce_join(val, separator=','),
            lambda: tf.constant(str(def_val)))
      return temp_vals

    fields = tf.map_fn(
        _parse_one_line_tf,
        lines,
        tf_types,
        parallel_iterations=64,
        name='parse_one_line_tf_map_fn')

    def _convert(x, target_type, name):
      if target_type in [DatasetConfig.FLOAT, DatasetConfig.DOUBLE]:
        return tf.string_to_number(
            x, tf.float32, name='convert_input_flt32/%s' % name)
      elif target_type == DatasetConfig.INT32:
        return tf.string_to_number(
            x, tf.int32, name='convert_input_int32/%s' % name)
      elif target_type == DatasetConfig.INT64:
        return tf.string_to_number(
            x, tf.int64, name='convert_input_int64/%s' % name)
      return x

    inputs = {
        self._input_fields[x]: _convert(fields[x], self._input_field_types[x],
                                        self._input_fields[x])
        for x in self._effective_fids
    }

    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    file_paths = []
    for x in self._input_path:
      file_paths.extend(tf.gfile.Glob(x))
    assert len(file_paths) > 0, 'match no files with %s' % self._input_path

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
          tf.data.TextLineDataset,
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
        self._parse_rtp, num_parallel_calls=num_parallel_calls)
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
