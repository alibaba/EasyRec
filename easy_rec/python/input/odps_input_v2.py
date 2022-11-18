# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util

try:
  import pai
except Exception:
  pass


class OdpsInputV2(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(OdpsInputV2,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

  def _parse_table(self, *fields):
    fields = list(fields)
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    assert len(
        self._input_path) > 0, 'match no files with %s' % self._input_path
    # check data_config are consistent with odps tables
    odps_util.check_input_field_and_types(self._data_config)

    selected_cols = ','.join(self._input_fields)
    record_defaults = [
        self.get_type_defaults(x, v)
        for x, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    if self._data_config.pai_worker_queue and \
        mode == tf.estimator.ModeKeys.TRAIN:
      logging.info('pai_worker_slice_num = %d' %
                   self._data_config.pai_worker_slice_num)
      work_queue = pai.data.WorkQueue(
          self._input_path,
          num_epochs=self.num_epochs,
          shuffle=self._data_config.shuffle,
          num_slices=self._data_config.pai_worker_slice_num * self._task_num)
      que_paths = work_queue.input_dataset()
      dataset = tf.data.TableRecordDataset(
          que_paths,
          record_defaults=record_defaults,
          selected_cols=selected_cols)
    elif self._data_config.chief_redundant and \
        mode == tf.estimator.ModeKeys.TRAIN:
      dataset = tf.data.TableRecordDataset(
          self._input_path,
          record_defaults=record_defaults,
          selected_cols=selected_cols,
          slice_id=max(self._task_index - 1, 0),
          slice_count=max(self._task_num - 1, 1))
    else:
      dataset = tf.data.TableRecordDataset(
          self._input_path,
          record_defaults=record_defaults,
          selected_cols=selected_cols,
          slice_id=self._task_index,
          slice_count=self._task_num)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if self._data_config.shuffle:
        dataset = dataset.shuffle(
            self._data_config.shuffle_buffer_size,
            seed=2020,
            reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size=self._data_config.batch_size)

    dataset = dataset.map(
        self._parse_table,
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
