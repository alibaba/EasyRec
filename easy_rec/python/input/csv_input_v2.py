# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.input.input import Input


class CSVInputV2(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(CSVInputV2,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    assert len(
        self._input_path) > 0, 'match no files with %s' % self._input_path

    if self._input_path[0].startswith('hdfs://'):
      # support hdfs input
      dataset = tf.data.TextLineDataset(self._input_path)
    else:
      num_epochs = self.num_epochs if mode == tf.estimator.ModeKeys.TRAIN else 1
      is_train = (mode == tf.estimator.ModeKeys.TRAIN)
      record_defaults = [
          self.get_type_defaults(x, v)
          for x, v in zip(self._input_field_types, self._input_field_defaults)
      ]
      dataset = tf.data.experimental.make_csv_dataset(
          self._input_path,
          self._data_config.batch_size,
          column_names=self._input_fields,
          field_delim=self._data_config.separator,
          column_defaults=record_defaults,
          header=False,
          num_epochs=num_epochs,
          shuffle=is_train and self._data_config.shuffle,
          num_parallel_reads=8,
          sloppy=is_train)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if self._data_config.chief_redundant:
        dataset = dataset.shard(
            max(self._task_num - 1, 1), max(self._task_index - 1, 0))
      else:
        dataset = dataset.shard(self._task_num, self._task_index)
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    dataset = dataset.map(map_func=self._preprocess, num_parallel_calls=8)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
