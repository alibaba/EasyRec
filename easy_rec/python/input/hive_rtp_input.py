# -*- coding: utf-8 -*-
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils.check_utils import check_split
from easy_rec.python.utils.hive_utils import HiveUtils
from easy_rec.python.utils.input_utils import string_to_number
from easy_rec.python.utils.tf_utils import get_tf_type


class HiveRTPInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False):
    super(HiveRTPInput, self).__init__(data_config, feature_config, input_path,
                                       task_index, task_num, check_mode)
    if input_path is None:
      return
    self._data_config = data_config
    self._feature_config = feature_config
    self._hive_config = input_path
    self._eval_batch_size = data_config.eval_batch_size
    self._fetch_size = self._hive_config.fetch_size

    self._num_epoch = data_config.num_epochs
    self._num_epoch_record = 1
    logging.info('input_fields: %s label_fields: %s' %
                 (','.join(self._input_fields), ','.join(self._label_fields)))

    self._rtp_separator = self._data_config.rtp_separator
    if not isinstance(self._rtp_separator, str):
      self._rtp_separator = self._rtp_separator.encode('utf-8')
    logging.info('rtp separator = %s' % self._rtp_separator)
    self._selected_cols = self._data_config.selected_cols \
        if self._data_config.selected_cols else None
    logging.info('select cols: %s' % self._selected_cols)

  def _parse_table(self, *fields):
    fields = list(fields)
    labels = fields[:-1]

    non_feature_cols = self._label_fields
    if self._selected_cols:
      cols = [c.strip() for c in self._selected_cols.split(',')]
      non_feature_cols = cols[:-1]
    # only for features, labels and sample_weight excluded
    record_types = [
        t for x, t in zip(self._input_fields, self._input_field_types)
        if x not in non_feature_cols
    ]
    feature_num = len(record_types)
    # assume that the last field is the generated feature column
    logging.info('field_delim = %s, input_field_name = %d' %
                 (self._data_config.separator, len(record_types)))

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

    fields = labels[len(self._label_fields):]
    for i in range(feature_num):
      field = string_to_number(tmp_fields[:, i], record_types[i], i)
      fields.append(field)

    field_keys = [x for x in self._input_fields if x not in self._label_fields]
    effective_fids = [field_keys.index(x) for x in self._effective_fields]
    inputs = {field_keys[x]: fields[x] for x in effective_fids}

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]
    return inputs

  def _get_batch_size(self, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
      return self._data_config.batch_size
    else:
      return self._eval_batch_size

  def _build(self, mode, params):
    # get input type
    list_type = [
        get_tf_type(t)
        for x, t in zip(self._input_fields, self._input_field_types)
        if x in self._label_fields
    ]
    list_type.append(tf.string)

    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    if self._selected_cols:
      cols = [c.strip() for c in self._selected_cols.split(',')]
      record_defaults = [
          self.get_type_defaults(t, v)
          for x, t, v in zip(self._input_fields, self._input_field_types,
                             self._input_field_defaults)
          if x in cols[:-1]
      ]
      logging.info('selected_cols: %s;' % (','.join(cols)))
    else:
      record_defaults = [
          self.get_type_defaults(t, v)
          for x, t, v in zip(self._input_fields, self._input_field_types,
                             self._input_field_defaults)
          if x in self._label_fields
      ]
    record_defaults.append('')
    logging.info('record_defaults: %s;' %
                 (','.join([str(i) for i in record_defaults])))

    sels = self._selected_cols if self._selected_cols else '*'
    _hive_read = HiveUtils(
        data_config=self._data_config,
        hive_config=self._hive_config,
        selected_cols=sels,
        record_defaults=record_defaults,
        mode=mode,
        task_index=self._task_index,
        task_num=self._task_num).hive_read

    dataset = tf.data.Dataset.from_generator(
        _hive_read,
        output_types=list_type,
        output_shapes=list_shapes,
        args=(self._hive_config.table_name,))

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(
          self._data_config.shuffle_buffer_size,
          seed=2022,
          reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      dataset = dataset.repeat(1)

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
