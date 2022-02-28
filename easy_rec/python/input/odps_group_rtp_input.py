# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.protos.dataset_pb2 import DatasetConfig

try:
  import pai
except Exception:
  pass


class OdpsGroupRTPInput(Input):
  """GroupRTPInput for parsing rtp fg input format on odps.

  Our new format(csv in table) of rtp output:
     label0, label1, grouped features, img, group_size
  For the feature column, features are separated by ,
     multiple values of one feature are separated by , such as:
     ...20beautysmartParis...
  The features column and labels are specified by data_config.selected_cols,
     columns are selected by names in the table
     such as: clk,features, the last selected column is features, the first
     selected columns are labels
  """

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1):
    super(OdpsGroupRTPInput, self).__init__(data_config, feature_config,
                                            input_path, task_index, task_num)
    logging.info('input_fields: %s label_fields: %s' %
                 (','.join(self._input_fields), ','.join(self._label_fields)))

  def _parse_table(self, *fields):
    fields = list(fields)

    label_record_defaults = [
        t for x, t, v in zip(self._input_fields, self._input_field_types,
                             self._input_field_defaults)
        if x in self._label_fields
    ]
    sample_fields = []
    # label
    for idx in range(len(label_record_defaults)):
      field = tf.string_split(
          fields[idx],
          self._data_config.group_sample_separator,
          skip_empty=False)
      if label_record_defaults[idx] in [DatasetConfig.INT32]:
        field = tf.string_to_number(field.values, tf.int32)
      elif label_record_defaults[idx] in [DatasetConfig.INT64]:
        field = tf.string_to_number(field.values, tf.int64)
      elif label_record_defaults[idx] in [DatasetConfig.FLOAT]:
        field = tf.string_to_number(field.values, tf.float32)
      elif field.values.dtype in [DatasetConfig.DOUBLE]:
        field = tf.string_to_number(field.values, tf.float64)
      else:
        field = field.values
      sample_fields.append(field)
    # features
    field = tf.string_split(
        fields[-3], self._data_config.group_sample_separator,
        skip_empty=False).values
    sample_fields.append(field)
    # pic_path
    sample_fields.append(fields[-2])
    # group_size
    sample_fields.append(fields[-1])

    labels = sample_fields[:-3]
    # only for features
    record_defaults = [
        self.get_type_defaults(t, v)
        for x, t, v in zip(self._input_fields, self._input_field_types,
                           self._input_field_defaults)
        if x not in self._label_fields
    ][:-2]
    logging.info('field_delim = %s' % self._data_config.separator)
    fields = tf.string_split(
        sample_fields[-3], self._data_config.separator, skip_empty=False)
    tmp_fields = tf.reshape(fields.values, [-1, len(record_defaults)])
    fields = []
    for i in range(len(record_defaults)):
      if type(record_defaults[i]) == int:
        fields.append(
            tf.string_to_number(
                tmp_fields[:, i], tf.int64, name='field_as_int_%d' % i))
      elif type(record_defaults[i]) in [float, np.float32, np.float64]:
        fields.append(
            tf.string_to_number(
                tmp_fields[:, i], tf.float32, name='field_as_flt_%d' % i))
      elif type(record_defaults[i]) in [str, type(u''), bytes]:
        fields.append(tmp_fields[:, i])
      elif type(record_defaults[i]) == bool:
        fields.append(
            tf.logical_or(
                tf.equal(tmp_fields[:, i], 'True'),
                tf.equal(tmp_fields[:, i], 'true')))
      else:
        assert 'invalid types: %s' % str(type(record_defaults[i]))

    field_keys = [x for x in self._input_fields if x not in self._label_fields]
    effective_fids = [field_keys.index(x) for x in self._effective_fields]
    inputs = {field_keys[x]: fields[x] for x in effective_fids[:-2]}

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]

    inputs[self._input_fields[-2]] = sample_fields[-2]
    inputs[self._input_fields[-1]] = sample_fields[-1]
    return inputs

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = [x for x in self._input_path.split(',')]

    # record_defaults = [
    #     self.get_type_defaults(t, v)
    #     for x, t, v in zip(self._input_fields, self._input_field_types,
    #                        self._input_field_defaults)
    #     if x in self._label_fields
    # ]
    record_defaults = [
        '' for x, t, v in zip(self._input_fields, self._input_field_types,
                              self._input_field_defaults)
        if x in self._label_fields
    ]

    # the actual features are in one single column
    record_defaults.append(
        self._data_config.separator.join([
            str(self.get_type_defaults(t, v))
            for x, t, v in zip(self._input_fields, self._input_field_types,
                               self._input_field_defaults)
            if x not in self._label_fields
        ]))
    # pic_path
    record_defaults.append('')
    # group_size
    record_defaults.append(np.int32(0))

    selected_cols = self._data_config.selected_cols \
        if self._data_config.selected_cols else None

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
