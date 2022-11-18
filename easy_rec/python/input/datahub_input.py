# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import traceback

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util
from easy_rec.python.utils.config_util import parse_time

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
except Exception:
  logging.warning(
      'DataHub is not installed[%s]. You can install it by: pip install pydatahub'
      % traceback.format_exc())
  DataHub = None


class DataHubInput(Input):
  """DataHubInput is used for online train."""

  def __init__(self,
               data_config,
               feature_config,
               datahub_config,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(DataHubInput,
          self).__init__(data_config, feature_config, '', task_index, task_num,
                         check_mode, pipeline_config)
    if DataHub is None:
      logging.error('please install datahub: ',
                    'pip install pydatahub ;Python 3.6 recommended')
    try:
      self._num_epoch = 0
      self._datahub_config = datahub_config
      if self._datahub_config is not None:
        akId = self._datahub_config.akId
        akSecret = self._datahub_config.akSecret
        endpoint = self._datahub_config.endpoint
        if not isinstance(akId, str):
          akId = akId.encode('utf-8')
          akSecret = akSecret.encode('utf-8')
          endpoint = endpoint.encode('utf-8')
        self._datahub = DataHub(akId, akSecret, endpoint)
      else:
        self._datahub = None
    except Exception as ex:
      logging.info('exception in init datahub: %s' % str(ex))
      pass
    self._offset_dict = {}
    if datahub_config:
      shard_result = self._datahub.list_shard(self._datahub_config.project,
                                              self._datahub_config.topic)
      shards = shard_result.shards
      self._all_shards = shards
      self._shards = [
          shards[i] for i in range(len(shards)) if (i % task_num) == task_index
      ]
      logging.info('all shards: %s' % str(self._shards))

      offset_type = datahub_config.WhichOneof('offset')
      if offset_type == 'offset_time':
        ts = parse_time(datahub_config.offset_time) * 1000
        for x in self._all_shards:
          ks = str(x.shard_id)
          cursor_result = self._datahub.get_cursor(self._datahub_config.project,
                                                   self._datahub_config.topic,
                                                   ks, CursorType.SYSTEM_TIME,
                                                   ts)
          logging.info('shard[%s] cursor = %s' % (ks, cursor_result))
          self._offset_dict[ks] = cursor_result.cursor
      elif offset_type == 'offset_info':
        self._offset_dict = json.loads(self._datahub_config.offset_info)
      else:
        self._offset_dict = {}

      self._dh_field_names = []
      self._dh_field_types = []
      topic_info = self._datahub.get_topic(
          project_name=self._datahub_config.project,
          topic_name=self._datahub_config.topic)
      for field in topic_info.record_schema.field_list:
        self._dh_field_names.append(field.name)
        self._dh_field_types.append(field.type.value)

      assert len(
          self._feature_fields) > 0, 'data_config.feature_fields are not set.'

      for x in self._feature_fields:
        assert x in self._dh_field_names, 'feature_field[%s] is not in datahub' % x

      # feature column ids in datahub schema
      self._dh_fea_ids = [
          self._dh_field_names.index(x) for x in self._feature_fields
      ]

      for x in self._label_fields:
        assert x in self._dh_field_names, 'label_field[%s] is not in datahub' % x

      if self._data_config.HasField('sample_weight'):
        x = self._data_config.sample_weight
        assert x in self._dh_field_names, 'sample_weight[%s] is not in datahub' % x

      self._read_cnt = 32

      if len(self._dh_fea_ids) > 1:
        self._filter_fea_func = lambda record: ''.join(
            [record.values[x]
             for x in self._dh_fea_ids]).split(chr(2))[1] == '-1024'
      else:
        dh_fea_id = self._dh_fea_ids[0]
        self._filter_fea_func = lambda record: record.values[dh_fea_id].split(
            self._data_config.separator)[1] == '-1024'

  def _parse_record(self, *fields):
    field_dict = {}
    fields = list(fields)

    def _dump_offsets():
      all_offsets = {
          x.shard_id: self._offset_dict[x.shard_id]
          for x in self._shards
          if x.shard_id in self._offset_dict
      }
      return json.dumps(all_offsets)

    field_dict[Input.DATA_OFFSET] = tf.py_func(_dump_offsets, [], dtypes.string)

    for x in self._label_fields:
      dh_id = self._dh_field_names.index(x)
      field_dict[x] = fields[dh_id]

    feature_inputs = self.get_feature_input_fields()
    # only for features, labels and sample_weight excluded
    record_types = [
        t for x, t in zip(self._input_fields, self._input_field_types)
        if x in feature_inputs
    ]
    feature_num = len(record_types)

    feature_fields = [
        fields[self._dh_field_names.index(x)] for x in self._feature_fields
    ]
    feature = feature_fields[0]
    for fea_id in range(1, len(feature_fields)):
      feature = feature + self._data_config.separator + feature_fields[fea_id]

    feature = tf.string_split(
        feature, self._data_config.separator, skip_empty=False)

    fields = tf.reshape(feature.values, [-1, feature_num])

    for fid in range(feature_num):
      field_dict[feature_inputs[fid]] = fields[:, fid]
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

  def restore(self, checkpoint_path):
    if checkpoint_path is None:
      return

    offset_path = checkpoint_path + '.offset'
    if not gfile.Exists(offset_path):
      return

    logging.info('will restore datahub offset from  %s' % offset_path)
    with gfile.GFile(offset_path, 'r') as fin:
      offset_dict = json.load(fin)
      for k in offset_dict:
        v = offset_dict[k]
        ks = str(k)
        if ks not in self._offset_dict or v > self._offset_dict[ks]:
          self._offset_dict[ks] = v

  def _is_data_empty(self, record):
    is_empty = True
    for fid in self._dh_fea_ids:
      if record.values[fid] is not None and len(record.values[fid]) > 0:
        is_empty = False
        break
    return is_empty

  def _dump_record(self, record):
    feas = []
    for fid in range(len(record.values)):
      if fid not in self._dh_fea_ids:
        feas.append(self._dh_field_names[fid] + ':' + str(record.values[fid]))
    return ';'.join(feas)

  def _datahub_generator(self):
    logging.info('start epoch[%d]' % self._num_epoch)
    self._num_epoch += 1

    try:
      self._datahub.wait_shards_ready(self._datahub_config.project,
                                      self._datahub_config.topic)
      topic_result = self._datahub.get_topic(self._datahub_config.project,
                                             self._datahub_config.topic)
      if topic_result.record_type != RecordType.TUPLE:
        logging.error('datahub topic type(%s) illegal' %
                      str(topic_result.record_type))
      record_schema = topic_result.record_schema

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
          cursor = self._offset_dict[shard_id]

        get_result = self._datahub.get_tuple_records(
            self._datahub_config.project, self._datahub_config.topic, shard_id,
            record_schema, cursor, self._read_cnt)
        count = get_result.record_count
        if count == 0:
          continue
        for row_id, record in enumerate(get_result.records):
          if self._is_data_empty(record):
            logging.warning('skip empty data record: %s' %
                            self._dump_record(record))
            continue
          if self._filter_fea_func is not None:
            if self._filter_fea_func(record):
              logging.warning('filter data record: %s' %
                              self._dump_record(record))
              continue
          yield tuple(list(record.values))
        if shard_id not in self._offset_dict or get_result.next_cursor > self._offset_dict[
            shard_id]:
          self._offset_dict[shard_id] = get_result.next_cursor
    except DatahubException as ex:
      logging.error('DatahubException: %s' % str(ex))

  def _build(self, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
      assert self._datahub is not None, 'datahub_train_input is not set'
    elif mode == tf.estimator.ModeKeys.EVAL:
      assert self._datahub is not None, 'datahub_eval_input is not set'

    # get input types
    list_types = [
        odps_util.odps_type_2_tf_type(x) for x in self._dh_field_types
    ]
    list_types = tuple(list_types)
    list_shapes = [
        tf.TensorShape([]) for x in range(0, len(self._dh_field_types))
    ]
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

    dataset = dataset.batch(self._data_config.batch_size)

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
