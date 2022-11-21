# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util

try:
  import pai
except Exception:
  pass


class OdpsInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(OdpsInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

  def _build(self, mode, params):
    # check data_config are consistent with odps tables
    odps_util.check_input_field_and_types(self._data_config)

    selected_cols = ','.join(self._input_fields)
    if self._data_config.chief_redundant and \
        mode == tf.estimator.ModeKeys.TRAIN:
      reader = tf.TableRecordReader(
          csv_delimiter=self._data_config.separator,
          selected_cols=selected_cols,
          slice_count=max(self._task_num - 1, 1),
          slice_id=max(self._task_index - 1, 0))
    else:
      reader = tf.TableRecordReader(
          csv_delimiter=self._data_config.separator,
          selected_cols=selected_cols,
          slice_count=self._task_num,
          slice_id=self._task_index)

    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    assert len(
        self._input_path) > 0, 'match no files with %s' % self._input_path

    if mode == tf.estimator.ModeKeys.TRAIN:
      if self._data_config.pai_worker_queue:
        work_queue = pai.data.WorkQueue(
            self._input_path,
            num_epochs=self.num_epochs,
            shuffle=self._data_config.shuffle,
            num_slices=self._data_config.pai_worker_slice_num * self._task_num)
        work_queue.add_summary()
        file_queue = work_queue.input_producer()
        reader = tf.TableRecordReader()
      else:
        file_queue = tf.train.string_input_producer(
            self._input_path,
            num_epochs=self.num_epochs,
            capacity=1000,
            shuffle=self._data_config.shuffle)
    else:
      file_queue = tf.train.string_input_producer(
          self._input_path, num_epochs=1, capacity=1000, shuffle=False)
    key, value = reader.read_up_to(file_queue, self._batch_size)

    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]
    fields = tf.decode_csv(
        value,
        record_defaults=record_defaults,
        field_delim=self._data_config.separator,
        name='decode_csv')

    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]

    fields = self._preprocess(inputs)

    features = self._get_features(fields)
    # import pai
    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = self._get_labels(fields)
      # features, labels = pai.data.prefetch(features=(features, labels),
      #                         capacity=self._prefetch_size, num_threads=2,
      #         closed_exception_types=(tuple([tf.errors.InternalError])))
      return features, labels
    else:
      # features = pai.data.prefetch(features=(features,),
      #                         capacity=self._prefetch_size, num_threads=2,
      #         closed_exception_types=(tuple([tf.errors.InternalError])))
      return features
