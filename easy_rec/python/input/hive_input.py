# -*- coding: utf-8 -*-
import logging
import numpy as np
import tensorflow as tf

try:
  from pyhive import hive
except ImportError:
  logging.warning('pyhive is not installed.')

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util


class TableInfo(object):

  def __init__(self,
               tablename,
               selected_cols,
               partition_kv,
               hash_fields,
               limit_num,
               batch_size=16,
               task_index=0,
               task_num=1,
               epoch=1):
    self.tablename = tablename
    self.selected_cols = selected_cols
    self.partition_kv = partition_kv
    self.hash_fields = hash_fields
    self.limit_num = limit_num
    self.task_index = task_index
    self.task_num = task_num
    self.batch_size = batch_size
    self.epoch = epoch

  def gen_sql(self):
    part = ''
    if self.partition_kv and len(self.partition_kv) > 0:
      res = []
      for k, v in self.partition_kv.items():
        res.append('{}={}'.format(k, v))
      part = ' '.join(res)
    sql = """select {}
        from {}""".format(self.selected_cols, self.tablename)
    assert self.hash_fields is not None, 'hash_fields must not be empty'
    fields = ['cast({} as string)'.format(key) for key in self.hash_fields.split(',')]
    str_fields = ','.join(fields)
    if not part:
      sql += """
        where hash(concat({}))%{}={}
        """.format(str_fields, self.task_num, self.task_index)
    else:
      sql += """
        where {} and hash(concat({}))%{}={}
        """.format(part, str_fields, self.task_num, self.task_index)
    if self.limit_num is not None and self.limit_num > 0:
      sql += ' limit {}'.format(self.limit_num)
    return sql


class HiveManager(object):

  def __init__(self,
               host,
               port,
               username,
               info,
               database='default'):
    self.host = host
    self.port = port
    self.username = username
    self.database = database
    self.info = info

  def __call__(self):
    conn = hive.Connection(
        host=self.host,
        port=self.port,
        username=self.username,
        database=self.database)
    cursor = conn.cursor()
    sql = self.info.gen_sql()
    res = []
    for ep in range(self.info.epoch):
      cursor.execute(sql)
      for result in cursor.fetchall():
        res.append(result)
        if len(res) == self.info.batch_size:
          yield res
          res = []
    pass


class HiveInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1):
    super(HiveInput, self).__init__(data_config, feature_config, input_path,
                                    task_index, task_num)
    self._hive_config = input_path
    self._num_epoch = data_config.num_epochs
    self._num_epoch_record = 1

  def _construct_table_info(self, table_name, hash_fields, limit_num):
    # sample_table/dt=2014-11-23/name=a
    segs = table_name.split('/')
    table_name = segs[0].strip()
    if len(segs) > 0:
      partition_kv = {i.split('=')[0]: i.split('=')[1] for i in segs[1:]}
    else:
      partition_kv = None
    selected_cols = ','.join(self._input_fields)
    table_info = TableInfo(table_name, selected_cols, partition_kv, hash_fields,
                           limit_num, self._data_config.batch_size,
                           self._task_index, self._task_num, self._num_epoch)
    return table_info

  def _construct_hive_connect(self):
    conn = hive.Connection(
        host=self._hive_config.host,
        port=self._hive_config.port,
        username=self._hive_config.username,
        database=self._hive_config.database)
    return conn

  def _parse_table(self, *fields):
    fields = list(fields)
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _hive_read(self):
    logging.info('start epoch[%d]' % self._num_epoch_record)
    self._num_epoch_record += 1
    talbe_names = [t for t in str(self._hive_config.table_name).split(',')]

    # check data_config are consistent with odps tables
    odps_util.check_input_field_and_types(self._data_config)

    record_defaults = [
        self.get_type_defaults(x, v)
        for x, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    for table_path in talbe_names:
      table_info = self._construct_table_info(table_path,
                                              self._hive_config.hash_fields,
                                              self._hive_config.limit_num)
      conn = self._construct_hive_connect()
      cursor = conn.cursor()
      sql = table_info.gen_sql()
      res = []
      cursor.execute(sql)

      batch_defaults = [
          np.array([x] * self._data_config.batch_size) for x in record_defaults
      ]

      row_id = 0
      batch_data_np = [x.copy() for x in batch_defaults]
      for result in cursor.fetchall():
        res.append(1)
        for col_id in range(len(record_defaults)):
          if result[col_id] not in ['', 'NULL', None]:
            batch_data_np[col_id][row_id] = result[col_id]
        if len(res) == self._data_config.batch_size:
          yield tuple(batch_data_np)
          res = []
          row_id = 0
          batch_data_np = [x.copy() for x in batch_defaults]
        else:
          row_id += 1

      if len(res) > 0:
        yield tuple(batch_data_np)
      conn.close()
    logging.info('finish epoch[%d]' % self._num_epoch_record)

  def _build(self, mode, params):
    # get input type
    list_type = [self.get_tf_type(x) for x in self._input_field_types]
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    # read odps tables

    dataset = tf.data.Dataset.from_generator(
        self._hive_read, output_types=list_type, output_shapes=list_shapes)

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
