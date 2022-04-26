# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import time

import numpy as np
import json
import tensorflow as tf
import traceback

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
  import urllib3
  urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
  logging.getLogger('datahub.account').setLevel(logging.INFO)
except Exception as ex:
  logging.warning(
      'DataHub is not installed[%s]. You can install it by: pip install pydatahub' % str(ex))
  DataHub = None

class DataHubInput(Input):
  """DataHubInput is used for online train."""


  def __init__(self,
               data_config,
               feature_config,
               datahub_config,
               task_index=0,
               task_num=1):
    super(DataHubInput, self).__init__(data_config, feature_config, '',
                                       task_index, task_num)

    try:
      self._num_epoch = 0
      self._datahub_config = datahub_config
      if self._datahub_config is not None:
        akId = self._datahub_config.akId
        akSecret = self._datahub_config.akSecret
        region = self._datahub_config.region
        if not isinstance(akId, str):
          akId = akId.encode('utf-8')
          akSecret = akSecret.encode('utf-8')
          region = region.encode('utf-8')
        self._datahub = DataHub(akId, akSecret, region)
      else:
        self._datahub = None
    except Exception as ex:
      logging.info('exception in init datahub: %s' % str(ex))
      pass
    self._offset_dict = {}
    if datahub_config:
      if self._datahub_config.offset_info:
        self._offset_dict = json.loads(self._datahub_config.offset_info)
      shard_result = self._datahub.list_shard(self._datahub_config.project,
                                              self._datahub_config.topic)
      shards = shard_result.shards
      self._shards = [shards[i] for i in range(len(shards)) if (i % task_num) == task_index]
      logging.info('all shards: %s' % str(self._shards))
      offset_dict = {}
      for x in self._shards:
        if x.shard_id in self._offset_dict:
          offset_dict[x.shard_id] = self._offset_dict[x.shard_id]
      self._offset_dict = offset_dict

  def _parse_record(self, *fields):
    fields = list(fields)
    field_dict = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      field_dict[self._input_fields[x]] = fields[x]
    field_dict[Input.DATA_OFFSET] = fields[-1]
    return field_dict

  def _preprocess(self, field_dict):
    output_dict = super(DataHubInput, self)._preprocess(field_dict)

    # append offset fields
    if Input.DATA_OFFSET in field_dict:
      output_dict[Input.DATA_OFFSET] = field_dict[Input.DATA_OFFSET]

    # for _get_features to include DATA_OFFSET
    if Input.DATA_OFFSET not in self._appended_fields: 
      self._appended_fields.append(Input.DATA_OFFSET)

    return output_dict

  def _datahub_generator(self):
    logging.info('start epoch[%d]' % self._num_epoch)
    self._num_epoch += 1
    odps_util.check_input_field_and_types(self._data_config)
    record_defaults = [
        self.get_type_defaults(x, v)
        for x, v in zip(self._input_field_types, self._input_field_defaults)
    ]
    batch_data = [
        np.asarray([x] * self._data_config.batch_size, order='C', dtype=object) 
        if isinstance(x, str) else
        np.array([x] * self._data_config.batch_size)
        for x in record_defaults
    ]
    batch_data.append(json.dumps(self._offset_dict))

    try:
      self._datahub.wait_shards_ready(self._datahub_config.project,
                                      self._datahub_config.topic)
      topic_result = self._datahub.get_topic(self._datahub_config.project,
                                             self._datahub_config.topic)
      if topic_result.record_type != RecordType.TUPLE:
        logging.error('datahub topic type(%s) illegal' % str(topic_result.record_type))
      record_schema = topic_result.record_schema

      batch_size = self._data_config.batch_size

      tid = 0
      while True:
        shard_id = self._shards[tid].shard_id
        tid += 1
        if tid >= len(self._shards):
          tid = 0
        if shard_id not in self._offset_dict:
          cursor_result = self._datahub.get_cursor(self._datahub_config.project,
                                                 self._datahub_config.topic,
                                                 shard_id, CursorType.OLDEST)
          cursor = cursor_result.cursor
        else:
          cursor = self._offset_dict[shard_id]['cursor']

        get_result = self._datahub.get_tuple_records(
            self._datahub_config.project, self._datahub_config.topic,
            shard_id, record_schema, cursor, batch_size)
        count = get_result.record_count
        if count == 0:
          continue
        time_offset = 0
        sequence_offset = 0
        for row_id, record in enumerate(get_result.records):
          if record.system_time > time_offset:
            time_offset = record.system_time
          if record.sequence > sequence_offset:
            sequence_offset = record.sequence 
          for col_id in range(len(record_defaults)):
            if record.values[col_id] not in ['', 'Null', 'null', 'NULL', None]:
              batch_data[col_id][row_id] = record.values[col_id]
            else:
              batch_data[col_id][row_id] = record_defaults[col_id]
        cursor = get_result.next_cursor
        self._offset_dict[shard_id] = {'sequence_offset': sequence_offset,
                                       'time_offset': time_offset, 
                                       'cursor': cursor
                                      } 
        batch_data[-1] = json.dumps(self._offset_dict)
        yield tuple(batch_data)
    except DatahubException as ex:
      logging.error('DatahubException: %s' % str(ex))

  def _build(self, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
      assert self._datahub is not None, "datahub_train_input is not set"
    elif mode == tf.estimator.ModeKeys.EVAL:
      assert self._datahub is not None, "datahub_eval_input is not set"

    # get input types
    list_types = [self.get_tf_type(x) for x in self._input_field_types]
    list_types.append(tf.string)
    list_types = tuple(list_types)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(self._input_field_types))]
    list_shapes.append(tf.TensorShape([]))
    list_shapes = tuple(list_shapes)
    # read datahub
    dataset = tf.data.Dataset.from_generator(
        self._datahub_generator,
        output_types=list_types,
        output_shapes=list_shapes)
    if mode == tf.estimator.ModeKeys.TRAIN:
      if self._data_config.shuffle:
        dataset = dataset.shuffle(
            self._data_config.shuffle_buffer_size,
            seed=2020,
            reshuffle_each_iteration=True)

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
