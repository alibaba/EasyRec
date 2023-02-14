# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime
import json
import logging
import time
import traceback

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
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
  # from datahub.exceptions import DatahubException
  from datahub.models import RecordType
  from datahub.models import CursorType
  from datahub.models.shard import ShardState
  import urllib3
  urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
  logging.getLogger('datahub.account').setLevel(logging.INFO)
except Exception:
  logging.warning(
      'DataHub is not installed[%s]. You can install it by: pip install pydatahub'
      % traceback.format_exc())
  DataHub = None

if tf.__version__ >= '2.0':
  ignore_errors = tf.data.experimental.ignore_errors()
  tf = tf.compat.v1
else:
  ignore_errors = tf.contrib.data.ignore_errors()


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
        self._max_conn_num = 8
        self._datahub = DataHub(
            akId, akSecret, endpoint, pool_maxsize=self._max_conn_num)
      else:
        self._datahub = None
    except Exception as ex:
      logging.info('exception in init datahub: %s' % str(ex))
      pass
    self._offset_dict = {}
    if datahub_config:
      shard_result = self._datahub.list_shard(self._datahub_config.project,
                                              self._datahub_config.topic)
      shards = [x for x in shard_result.shards if x.state == ShardState.ACTIVE]
      self._all_shards = shards

      if self._data_config.chief_redundant and self._task_num > 1:
        if task_index == 0:
          self._shards = [shards[0]]
        else:
          task_num -= 1
          task_index -= 1
          self._shards = [
              shards[i]
              for i in range(len(shards))
              if (i % task_num) == task_index
          ]
      else:
        self._shards = [
            shards[i]
            for i in range(len(shards))
            if (i % task_num) == task_index
        ]

      logging.info('all_shards[len=%d]: %s task_shards[len=%d]: %s' %
                   (len(self._all_shards), str(
                       self._all_shards), len(self._shards), str(self._shards)))

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

      self._dh_lbl_ids = [
          self._dh_field_names.index(x) for x in self._label_fields
      ]

      if self._data_config.HasField('sample_weight'):
        x = self._data_config.sample_weight
        assert x in self._dh_field_names, 'sample_weight[%s] is not in datahub' % x

      self._read_cnt = 512
      self._log_every_cnts = self._data_config.batch_size * 16
      self._max_retry = 8

      # record shard read cnt
      self._shard_read_cnt = {}
      self._shard_cursor_seq = {}
      self._last_log_cnt = {}

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

    field_dict[Input.DATA_OFFSET] = script_ops.py_func(_dump_offsets, [],
                                                       dtypes.string)

    for x, dh_id in zip(self._label_fields, self._dh_lbl_ids):
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

    feature = string_ops.string_split(
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

  def _datahub_generator_for_redundant_chief(self):
    logging.info('start chief redundant epoch[%d]' % self._num_epoch)
    self._num_epoch += 1

    self._datahub.wait_shards_ready(self._datahub_config.project,
                                    self._datahub_config.topic)
    topic_result = self._datahub.get_topic(self._datahub_config.project,
                                           self._datahub_config.topic)
    if topic_result.record_type != RecordType.TUPLE:
      logging.error('datahub topic type(%s) illegal' %
                    str(topic_result.record_type))
    record_schema = topic_result.record_schema
    shard = self._shards[0]
    shard_id = shard.shard_id

    if shard_id not in self._offset_dict:
      cursor_result = self._datahub.get_cursor(self._datahub_config.project,
                                               self._datahub_config.topic,
                                               shard_id, CursorType.OLDEST)
      cursor = cursor_result.cursor
    else:
      cursor = self._offset_dict[shard_id]
    all_records = []
    while len(all_records) <= self._batch_size * 8:
      try:
        get_result = self._datahub.get_tuple_records(
            self._datahub_config.project, self._datahub_config.topic, shard_id,
            record_schema, cursor, self._read_cnt)
        for row_id, record in enumerate(get_result.records):
          if self._is_data_empty(record):
            logging.warning('skip empty data record: %s' %
                            self._dump_record(record))
            continue
          all_records.append(tuple(record.values))
      except Exception as ex:
        logging.warning(
            'get_tuple_records exception: shard_id=%s cursor=%s read_cnt=%d exception:%s traceback:%s'
            %
            (shard_id, cursor, self._read_cnt, str(ex), traceback.format_exc()))
    sid = 0
    while True:
      if sid >= len(all_records):
        sid = 0
      yield all_records[sid]
      sid += 1

  def _datahub_generator(self, part_id, part_num):
    avg_num = len(self._shards) / part_num
    res_num = len(self._shards) % part_num
    start_id = avg_num * part_id + min(part_id, res_num)
    end_id = avg_num * (part_id + 1) + min(part_id + 1, res_num)

    thread_shards = self._shards[start_id:end_id]

    logging.info(
        'start generator[part_id=%d][part_num=%d][shard_num=%d][thread_shard_num=%d:%d]'
        % (part_id, part_num, len(self._shards), start_id, end_id))

    try:
      self._datahub.wait_shards_ready(self._datahub_config.project,
                                      self._datahub_config.topic)
      topic_result = self._datahub.get_topic(self._datahub_config.project,
                                             self._datahub_config.topic)
      if topic_result.record_type != RecordType.TUPLE:
        logging.error('datahub topic type(%s) illegal' %
                      str(topic_result.record_type))
      record_schema = topic_result.record_schema

      try:
        iter_id = 0
        while True:
          shard_id = thread_shards[iter_id].shard_id
          iter_id += 1
          if iter_id >= len(thread_shards):
            iter_id = 0
          if shard_id not in self._offset_dict:
            cursor_result = self._datahub.get_cursor(
                self._datahub_config.project, self._datahub_config.topic,
                shard_id, CursorType.OLDEST)
            cursor = cursor_result.cursor
          else:
            cursor = self._offset_dict[shard_id]

          max_retry = self._max_retry
          get_result = None
          while max_retry > 0:
            try:
              get_result = self._datahub.get_tuple_records(
                  self._datahub_config.project, self._datahub_config.topic,
                  shard_id, record_schema, cursor, self._read_cnt)
              break
            except Exception as ex:
              logging.warning(
                  'get_tuple_records exception: shard_id=%s cursor=%s read_cnt=%d exception:%s traceback:%s'
                  % (shard_id, cursor, self._read_cnt, str(ex),
                     traceback.format_exc()))
            max_retry -= 1
          if get_result is None:
            logging.error('failed to get_tuple_records after max_retry=%d' %
                          self._max_retry)
            raise RuntimeError(
                'failed to get_tuple_records after max_retry=%d' %
                self._max_retry)
          count = get_result.record_count
          if count == 0:
            if get_result.next_cursor > cursor:
              self._offset_dict[shard_id] = get_result.next_cursor
            else:
              # avoid too frequent access to datahub server
              time.sleep(0.1)
            continue
          self._shard_cursor_seq[shard_id] = get_result.start_seq + (count - 1)
          for row_id, record in enumerate(get_result.records):
            if self._is_data_empty(record):
              logging.warning('skip empty data record: %s' %
                              self._dump_record(record))
              continue
            yield tuple(record.values)
          if shard_id not in self._offset_dict or get_result.next_cursor > self._offset_dict[
              shard_id]:
            self._offset_dict[shard_id] = get_result.next_cursor
          self._update_counter(shard_id, count)
      except Exception as ex:
        logging.error('fetch_sample thread[shard_id=%s] fail: %s %s' %
                      (shard_id, str(ex), traceback.format_exc()))
    except Exception as ex:
      logging.error('_datahub_generator exception: %s %s' %
                    (str(ex), traceback.format_exc()))

  def _update_counter(self, shard_id, count):
    if count == 0:
      return
    if shard_id not in self._shard_read_cnt:
      self._shard_read_cnt[shard_id] = count
      self._log_shard(shard_id)
    else:
      self._shard_read_cnt[shard_id] += count
      tmp_cnt = self._last_log_cnt.get(shard_id, 0)
      if self._shard_read_cnt[shard_id] - tmp_cnt > self._log_every_cnts:
        self._log_shard(shard_id)
        self._last_log_cnt[shard_id] = self._shard_read_cnt[shard_id]

  def _log_shard(self, shard_id):
    if shard_id not in self._shard_cursor_seq:
      return
    tmp_seq = self._shard_cursor_seq[shard_id]
    if tmp_seq < 0:
      return
    cursor_result = self._datahub.get_cursor(
        self._datahub_config.project,
        self._datahub_config.topic,
        shard_id,
        CursorType.SEQUENCE,
        param=tmp_seq)
    ts = cursor_result.record_time / 1000.0
    ts_s = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    logging.info('shard[%s]: cursor=%s sequence=%d ts=%.3f datetime=%s' %
                 (shard_id, cursor_result.cursor, tmp_seq, ts, ts_s))

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
    if self._data_config.chief_redundant and self._task_num > 1 and self._task_index == 0:
      dataset = tf.data.Dataset.from_generator(
          self._datahub_generator_for_redundant_chief,
          output_types=list_types,
          output_shapes=list_shapes)
    else:
      split_num = min(self._max_conn_num, len(self._shards))
      dataset = tf.data.Dataset.from_tensor_slices(
          np.arange(split_num)).interleave(
              lambda x: tf.data.Dataset.from_generator(
                  self._datahub_generator,
                  output_types=list_types,
                  output_shapes=list_shapes,
                  args=[x, split_num]),
              cycle_length=len(self._shards),
              num_parallel_calls=len(self._shards))
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
    if self._data_config.ignore_error:
      dataset = dataset.apply(ignore_errors)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
