# -*- coding: utf-8 -*-
import logging

import numpy as np
import tensorflow as tf

try:
  from pyhive import hive
except ImportError:
  logging.warning('pyhive is not installed.')


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
    fields = [
        'cast({} as string)'.format(key) for key in self.hash_fields.split(',')
    ]
    str_fields = ','.join(fields)
    if not part:
      sql += """
    where abs(hash(concat({})))%{}={}
    """.format(str_fields, self.task_num, self.task_index)
    else:
      sql += """
    where {} and abs(hash(concat({})))%{}={}
    """.format(part, str_fields, self.task_num, self.task_index)
    if self.limit_num is not None and self.limit_num > 0:
      sql += ' limit {}'.format(self.limit_num)
    return sql


class HiveUtils(object):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               hive_config,
               mode,
               selected_cols='',
               record_defaults=[],
               task_index=0,
               task_num=1):

    self._data_config = data_config
    self._hive_config = hive_config
    self._eval_batch_size = data_config.eval_batch_size
    self._fetch_size = self._hive_config.fetch_size
    self._this_batch_size = self._get_batch_size(mode)

    self._num_epoch = data_config.num_epochs
    self._num_epoch_record = 0
    self._task_index = task_index
    self._task_num = task_num
    self._selected_cols = selected_cols
    self._record_defaults = record_defaults

  def _construct_table_info(self, table_name, hash_fields, limit_num):
    # sample_table/dt=2014-11-23/name=a
    segs = table_name.split('/')
    table_name = segs[0].strip()
    if len(segs) > 0:
      partition_kv = {i.split('=')[0]: i.split('=')[1] for i in segs[1:]}
    else:
      partition_kv = None

    table_info = TableInfo(table_name, self._selected_cols, partition_kv,
                           hash_fields, limit_num, self._data_config.batch_size,
                           self._task_index, self._task_num, self._num_epoch)
    return table_info

  def _construct_hive_connect(self):
    conn = hive.Connection(
        host=self._hive_config.host,
        port=self._hive_config.port,
        username=self._hive_config.username,
        database=self._hive_config.database)
    return conn

  def _get_batch_size(self, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
      return self._data_config.batch_size
    else:
      return self._eval_batch_size

  def hive_read(self, input_path):
    logging.info('start epoch[%d]' % self._num_epoch_record)
    self._num_epoch_record += 1
    if type(input_path) != type(str):
      input_path = input_path.decode('utf-8')

    for table_path in input_path.split(','):
      table_info = self._construct_table_info(table_path,
                                              self._hive_config.hash_fields,
                                              self._hive_config.limit_num)
      batch_size = self._this_batch_size
      batch_defaults = []
      for x in self._record_defaults:
        if isinstance(x, str):
          batch_defaults.append(np.array([x] * batch_size, dtype='S2000'))
        else:
          batch_defaults.append(np.array([x] * batch_size))

      row_id = 0
      batch_data_np = [x.copy() for x in batch_defaults]

      conn = self._construct_hive_connect()
      cursor = conn.cursor()
      sql = table_info.gen_sql()
      cursor.execute(sql)

      while True:
        data = cursor.fetchmany(size=self._fetch_size)
        if len(data) == 0:
          break
        for rows in data:
          for col_id in range(len(self._record_defaults)):
            if rows[col_id] not in ['', 'NULL', None]:
              batch_data_np[col_id][row_id] = rows[col_id]
            else:
              batch_data_np[col_id][row_id] = batch_defaults[col_id][row_id]
          row_id += 1

          if row_id >= batch_size:
            yield tuple(batch_data_np)
            row_id = 0

      if row_id > 0:
        yield tuple([x[:row_id] for x in batch_data_np])
      cursor.close()
      conn.close()
    logging.info('finish epoch[%d]' % self._num_epoch_record)

  def hive_read_line(self, input_path, hash_fields, limit_num=None):
    table_info = self._construct_table_info(input_path, hash_fields, limit_num)
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    sql = table_info.gen_sql()
    cursor.execute(sql)

    while True:
      data = cursor.fetchmany(size=1)
      if len(data) == 0:
        break
      yield data

    cursor.close()
    conn.close()

  def run_sql(self, sql):
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    return data

  def is_table_or_partition_exist(self,
                                  table_name,
                                  partition_name=None,
                                  partition_val=None):
    if partition_name and partition_val:
      sql = 'show partitions %s partition(%s=%s)' % (table_name, partition_name, partition_val)
      try:
        res = self.run_sql(sql)
        if not res:
          return False
        else:
          return True
      except:
        return False

    else:
      sql = 'desc %s' % table_name
      try:
        self.run_sql(sql)
        return True
      except:
        return False

  def get_all_cols(self, input_path):
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    sql = 'desc %s' % input_path.split('/')[0]
    cursor.execute(sql)
    data = cursor.fetchmany()
    col_names = []
    cols_types = []
    for col in data:
      col_name = col[0].strip()
      if col_name and (not col_name.startswith('#')) and (col_name
                                                          not in col_names):
        col_names.append(col_name)
        cols_types.append(col[1].strip())

    return ','.join(col_names), ','.join(cols_types)
