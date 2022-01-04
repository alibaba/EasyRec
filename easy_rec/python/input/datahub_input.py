# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import time

import numpy as np
import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util

try:
  import common_io
except Exception:
  common_io = None
try:
  from datahub import DataHub
  from datahub.exceptions import DatahubException
  from datahub.models import RecordType
  from datahub.models import CursorType
except Exception:
  logging.warning(
      'DataHub is not installed. You can install it by: pip install pydatahub')
  DataHub = None


class DataHubInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               datahub_config,
               task_index=0,
               task_num=1):
    super(DataHubInput, self).__init__(data_config, feature_config, '',
                                       task_index, task_num)
    if DataHub is None:
      logging.error('please install datahub: ',
                    'pip install pydatahub ;Python 3.6 recommended')
    try:
      self._datahub_config = datahub_config
      if self._datahub_config is None:
        pass
      self._datahub = DataHub(self._datahub_config.akId,
                              self._datahub_config.akSecret,
                              self._datahub_config.region)
      self._num_epoch = 0
    except Exception as ex:
      logging.info('exception in init datahub:', str(ex))
      pass

  def _parse_record(self, *fields):
    fields = list(fields)
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _datahub_generator(self):
    logging.info('start epoch[%d]' % self._num_epoch)
    self._num_epoch += 1
    odps_util.check_input_field_and_types(self._data_config)
    record_defaults = [
        self.get_type_defaults(x, v)
        for x, v in zip(self._input_field_types, self._input_field_defaults)
    ]
    batch_defaults = [
        np.array([x] * self._data_config.batch_size) for x in record_defaults
    ]
    try:
      self._datahub.wait_shards_ready(self._datahub_config.project,
                                      self._datahub_config.topic)
      topic_result = self._datahub.get_topic(self._datahub_config.project,
                                             self._datahub_config.topic)
      if topic_result.record_type != RecordType.TUPLE:
        logging.error('topic type illegal !')
      record_schema = topic_result.record_schema
      shard_result = self._datahub.list_shard(self._datahub_config.project,
                                              self._datahub_config.topic)
      shards = shard_result.shards
      for shard in shards:
        shard_id = shard._shard_id
        cursor_result = self._datahub.get_cursor(self._datahub_config.project,
                                                 self._datahub_config.topic,
                                                 shard_id, CursorType.OLDEST)
        cursor = cursor_result.cursor
        limit = self._data_config.batch_size
        while True:
          get_result = self._datahub.get_tuple_records(
              self._datahub_config.project, self._datahub_config.topic,
              shard_id, record_schema, cursor, limit)
          batch_data_np = [x.copy() for x in batch_defaults]
          for row_id, record in enumerate(get_result.records):
            for col_id in range(len(record_defaults)):
              if record.values[col_id] not in ['', 'Null', None]:
                batch_data_np[col_id][row_id] = record.values[col_id]
          yield tuple(batch_data_np)
          if 0 == get_result.record_count:
            time.sleep(1)
          cursor = get_result.next_cursor
    except DatahubException as e:
      logging.error(e)

  def _build(self, mode, params):
    # get input type
    list_type = [self.get_tf_type(x) for x in self._input_field_types]
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)
    # read datahub
    dataset = tf.data.Dataset.from_generator(
        self._datahub_generator,
        output_types=list_type,
        output_shapes=list_shapes)
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(
          self._data_config.shuffle_buffer_size,
          seed=2020,
          reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      dataset = dataset.repeat(1)
    dataset = dataset.map(
        self._parse_record,
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
