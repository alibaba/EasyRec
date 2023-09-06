# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.ops.gen_str_avx_op import str_split_by_chr
from easy_rec.python.utils.check_utils import check_split
from easy_rec.python.utils.check_utils import check_string_to_number
from easy_rec.python.utils.input_utils import string_to_number
from easy_rec.python.utils.tf_utils import get_tf_type

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class RTPInput(Input):
  """RTPInput for parsing rtp fg new input format.

  Our new format(csv in csv) of rtp output:
     label0, item_id, ..., user_id, features
  here the separator(,) could be specified by data_config.rtp_separator
  For the feature column, features are separated by ,
     multiple values of one feature are separated by , such as:
     ...20beautysmartParis...
  The features column and labels are specified by data_config.selected_cols,
     columns are selected by indices as our csv file has no header,
     such as: 0,1,4, means the 4th column is features, the 1st and 2nd
     columns are labels
  """

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(RTPInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    logging.info('input_fields: %s label_fields: %s' %
                 (','.join(self._input_fields), ','.join(self._label_fields)))
    self._rtp_separator = self._data_config.rtp_separator
    if not isinstance(self._rtp_separator, str):
      self._rtp_separator = self._rtp_separator.encode('utf-8')
    self._selected_cols = [
        int(x) for x in self._data_config.selected_cols.split(',')
    ]
    self._num_cols = -1
    self._feature_col_id = self._selected_cols[-1]
    logging.info('rtp separator = %s' % self._rtp_separator)

  def _parse_csv(self, line):
    record_defaults = ['' for i in range(self._num_cols)]

    # the actual features are in one single column
    record_defaults[self._feature_col_id] = self._data_config.separator.join([
        str(self.get_type_defaults(t, v))
        for x, t, v in zip(self._input_fields, self._input_field_types,
                           self._input_field_defaults)
        if x not in self._label_fields
    ])

    check_list = [
        tf.py_func(
            check_split, [line, self._rtp_separator,
                          len(record_defaults)],
            Tout=tf.bool)
    ] if self._check_mode else []
    with tf.control_dependencies(check_list):
      fields = tf.string_split(line, self._rtp_separator, skip_empty=False)

    fields = tf.reshape(fields.values, [-1, len(record_defaults)])

    labels = []
    for idx, x in enumerate(self._selected_cols[:-1]):
      field = fields[:, x]
      fname = self._input_fields[idx]
      ftype = self._input_field_types[idx]
      tf_type = get_tf_type(ftype)
      if field.dtype in [tf.string]:
        check_list = [
            tf.py_func(check_string_to_number, [field, fname], Tout=tf.bool)
        ] if self._check_mode else []
        with tf.control_dependencies(check_list):
          field = tf.string_to_number(field, tf_type)
      labels.append(field)

    # only for features, labels excluded
    record_types = [
        t for x, t in zip(self._input_fields, self._input_field_types)
        if x not in self._label_fields
    ]
    # assume that the last field is the generated feature column
    print('field_delim = %s' % self._data_config.separator)
    feature_str = fields[:, self._feature_col_id]
    check_list = [
        tf.py_func(
            check_split,
            [feature_str, self._data_config.separator,
             len(record_types)],
            Tout=tf.bool)
    ] if self._check_mode else []
    with tf.control_dependencies(check_list):
      fields = str_split_by_chr(
          feature_str, self._data_config.separator, skip_empty=False)
    tmp_fields = tf.reshape(fields.values, [-1, len(record_types)])
    rtp_record_defaults = [
        str(self.get_type_defaults(t, v))
        for x, t, v in zip(self._input_fields, self._input_field_types,
                           self._input_field_defaults)
        if x not in self._label_fields
    ]
    fields = []
    for i in range(len(record_types)):
      field = string_to_number(tmp_fields[:, i], record_types[i],
                               rtp_record_defaults[i], i)
      fields.append(field)

    field_keys = [x for x in self._input_fields if x not in self._label_fields]
    effective_fids = [field_keys.index(x) for x in self._effective_fields]
    inputs = {field_keys[x]: fields[x] for x in effective_fids}

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]
    return inputs

  def _build(self, mode, params):
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    file_paths = []
    for x in self._input_path:
      file_paths.extend(tf.gfile.Glob(x))
    assert len(file_paths) > 0, 'match no files with %s' % self._input_path

    # try to figure out number of fields from one file
    with tf.gfile.GFile(file_paths[0], 'r') as fin:
      num_lines = 0
      for line_str in fin:
        line_tok = line_str.strip().split(self._rtp_separator)
        if self._num_cols != -1:
          assert self._num_cols == len(line_tok), \
              'num selected cols is %d, not equal to %d, current line is: %s, please check rtp_separator and data.' % \
              (self._num_cols, len(line_tok), line_str)
        self._num_cols = len(line_tok)
        num_lines += 1
        if num_lines > 10:
          break
    logging.info('num selected cols = %d' % self._num_cols)

    record_defaults = [
        self.get_type_defaults(t, v)
        for x, t, v in zip(self._input_fields, self._input_field_types,
                           self._input_field_defaults)
        if x in self._label_fields
    ]

    # the features are in one single column
    record_defaults.append(
        self._data_config.separator.join([
            str(self.get_type_defaults(t, v))
            for x, t, v in zip(self._input_fields, self._input_field_types,
                               self._input_field_defaults)
            if x not in self._label_fields
        ]))

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

    dataset = dataset.batch(batch_size=self._data_config.batch_size)

    dataset = dataset.map(
        self._parse_csv,
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
