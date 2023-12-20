# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import sys

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util
from easy_rec.python.utils.tf_utils import get_tf_type

try:
  import common_io
except Exception:
  common_io = None


class OdpsInputV3(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(OdpsInputV3,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    self._num_epoch = 0
    if common_io is None:
      logging.error("""please install common_io pip install
                    https://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/common_io-0.1.0-cp37-cp37m-linux_x86_64.whl"""
                    )
      sys.exit(1)

  def _parse_table(self, *fields):
    fields = list(fields)
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _odps_read(self):
    logging.info('start epoch[%d]' % self._num_epoch)
    self._num_epoch += 1
    if type(self._input_path) != list:
      self._input_path = self._input_path.split(',')
    assert len(
        self._input_path) > 0, 'match no files with %s' % self._input_path

    # check data_config are consistent with odps tables
    odps_util.check_input_field_and_types(self._data_config)

    record_defaults = [
        self.get_type_defaults(x, v)
        for x, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    selected_cols = ','.join(self._input_fields)
    for table_path in self._input_path:
      reader = common_io.table.TableReader(
          table_path,
          selected_cols=selected_cols,
          slice_id=self._task_index,
          slice_count=self._task_num)
      total_records_num = reader.get_row_count()
      batch_num = int(total_records_num / self._data_config.batch_size)
      res_num = total_records_num - batch_num * self._data_config.batch_size
      batch_defaults = [
          [x] * self._data_config.batch_size for x in record_defaults
      ]
      for batch_id in range(batch_num):
        batch_data_np = [x.copy() for x in batch_defaults]
        for row_id, one_data in enumerate(
            reader.read(self._data_config.batch_size)):
          for col_id in range(len(record_defaults)):
            if one_data[col_id] not in ['', 'NULL', None]:
              batch_data_np[col_id][row_id] = one_data[col_id]
        yield tuple(batch_data_np)
      if res_num > 0:
        batch_data_np = [x[:res_num] for x in batch_defaults]
        for row_id, one_data in enumerate(reader.read(res_num)):
          for col_id in range(len(record_defaults)):
            if one_data[col_id] not in ['', 'NULL', None]:
              batch_data_np[col_id][row_id] = one_data[col_id]
        yield tuple(batch_data_np)
      reader.close()
    logging.info('finish epoch[%d]' % self._num_epoch)

  def _build(self, mode, params):
    # get input type
    list_type = [get_tf_type(x) for x in self._input_field_types]
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    # read odps tables
    dataset = tf.data.Dataset.from_generator(
        self._odps_read, output_types=list_type, output_shapes=list_shapes)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(
          self._data_config.shuffle_buffer_size,
          seed=2020,
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
