# -*- coding: utf-8 -*-
import logging

try:
  from pyhive import hive
  from pyhive.exc import ProgrammingError
except ImportError:
  logging.warning('pyhive is not installed.')


class TableInfo(object):

  def __init__(self, tablename, selected_cols, partition_kv, limit_num):
    self.tablename = tablename
    self.selected_cols = selected_cols
    self.partition_kv = partition_kv
    self.limit_num = limit_num

  def gen_sql(self):
    part = ''
    if self.partition_kv and len(self.partition_kv) > 0:
      res = []
      for k, v in self.partition_kv.items():
        res.append('{}={}'.format(k, v))
      part = ' '.join(res)
    sql = """select {}
    from {}""".format(self.selected_cols, self.tablename)

    if part:
      sql += """
    where {}
    """.format(part)
    if self.limit_num is not None and self.limit_num > 0:
      sql += ' limit {}'.format(self.limit_num)
    return sql


class HiveUtils(object):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               hive_config,
               selected_cols='',
               record_defaults=[],
               task_index=0,
               task_num=1):

    self._data_config = data_config
    self._hive_config = hive_config

    self._num_epoch = data_config.num_epochs
    self._num_epoch_record = 0
    self._task_index = task_index
    self._task_num = task_num
    self._selected_cols = selected_cols
    self._record_defaults = record_defaults

  def _construct_table_info(self, table_name, limit_num):
    # sample_table/dt=2014-11-23/name=a
    segs = table_name.split('/')
    table_name = segs[0].strip()
    if len(segs) > 0:
      partition_kv = {i.split('=')[0]: i.split('=')[1] for i in segs[1:]}
    else:
      partition_kv = None

    table_info = TableInfo(table_name, self._selected_cols, partition_kv,
                           limit_num)
    return table_info

  def _construct_hive_connect(self):
    conn = hive.Connection(
        host=self._hive_config.host,
        port=self._hive_config.port,
        username=self._hive_config.username,
        database=self._hive_config.database)
    return conn

  def hive_read_line(self, input_path, limit_num=None):
    table_info = self._construct_table_info(input_path, limit_num)
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

  def hive_read_lines(self, input_path, batch_size, limit_num=None):
    table_info = self._construct_table_info(input_path, limit_num)
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    sql = table_info.gen_sql()
    cursor.execute(sql)

    while True:
      data = cursor.fetchmany(size=batch_size)
      if len(data) == 0:
        break
      yield data

    cursor.close()
    conn.close()

  def run_sql(self, sql):
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    cursor.execute(sql)
    try:
      data = cursor.fetchall()
    except ProgrammingError:
      data = []
    return data

  def is_table_or_partition_exist(self,
                                  table_name,
                                  partition_name=None,
                                  partition_val=None):
    if partition_name and partition_val:
      sql = 'show partitions %s partition(%s=%s)' % (table_name, partition_name,
                                                     partition_val)
      try:
        res = self.run_sql(sql)
        if not res:
          return False
        else:
          return True
      except:  # noqa: E722
        return False

    else:
      sql = 'desc %s' % table_name
      try:
        self.run_sql(sql)
        return True
      except:  # noqa: E722
        return False

  def get_table_location(self, input_path):
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    partition = ''
    if len(input_path.split('/')) == 2:
      table_name, partition = input_path.split('/')
      partition += '/'
    else:
      table_name = input_path
    sql = 'desc formatted %s' % table_name
    cursor.execute(sql)
    data = cursor.fetchmany()
    for line in data:
      if line[0].startswith('Location'):
        return line[1].strip() + '/' + partition
    return None

  def get_all_cols(self, input_path):
    conn = self._construct_hive_connect()
    cursor = conn.cursor()
    sql = 'desc %s' % input_path.split('/')[0]
    cursor.execute(sql)
    data = cursor.fetchmany()
    col_names = []
    cols_types = []
    pt_name = ''
    if len(input_path.split('/')) == 2:
      pt_name = input_path.split('/')[1].split('=')[0]

    for col in data:
      col_name = col[0].strip()
      if col_name and (not col_name.startswith('#')) and (col_name
                                                          not in col_names):
        if col_name != pt_name:
          col_names.append(col_name)
          cols_types.append(col[1].strip())

    return col_names, cols_types
