# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import time
import traceback

import oss2

try:
  from datahub import DataHub
  from datahub.exceptions import InvalidOperationException
  from datahub.exceptions import ResourceExistException
  # from datahub.exceptions import LimitExceededException
  # from datahub.exceptions import ResourceNotFoundException
  # from datahub.models import BlobRecord
  # from datahub.models import CursorType
  from datahub.models import FieldType
  from datahub.models import RecordSchema
  from datahub.models import RecordType
  from datahub.models import TupleRecord
except Exception:
  logging.error(
      'DataHub is not installed, please installed it by: pip install pydatahub')
  DataHub = None

try:
  from odps import ODPS
  from odps.df import DataFrame
except Exception:
  ODPS = None
  DataFrame = None


class OdpsOSSConfig:

  def __init__(self, script_path='./samples/odps_script'):
    self.time_stamp = int(time.time())
    temp_dir = os.environ.get('TMPDIR', '/tmp')
    self.exp_dir = 'easy_rec_odps_test_%d' % self.time_stamp
    self.temp_dir = os.path.join(temp_dir, self.exp_dir)
    self.log_dir = os.path.join(self.temp_dir, 'logs/')

    # public buckets with readyonly access
    self.ali_bucket_endpoint = 'http://oss-cn-beijing.aliyuncs.com'
    self.ali_bucket_name = 'easyrec'
    self.script_path = script_path
    # read only access
    self.ali_oss_key = os.environ['ALI_OSS_KEY']
    self.ali_oss_secret = os.environ['ALI_OSS_SEC']

    self.oss_key = ''
    self.oss_secret = ''
    self.endpoint = ''
    self.arn = 'acs:ram::xxx:role/aliyunodpspaidefaultrole'
    self.bucket_name = ''

    self.odpscmd_path = os.environ.get('ODPS_CMD_PATH', 'odpscmd')
    self.odps_config_path = ''

    self.project_name = ''

    self.dh_id = ''
    self.dh_key = ''

    self.dh_endpoint = 'https://dh-cn-beijing.aliyuncs.com'
    self.dh_topic = 'easy_rec_test'
    self.dh_project = 'easy_rec_test'

    self.odps_endpoint = ''

    self.dh = None

    # default to algo_public
    self.algo_project = None
    self.algo_res_project = None
    self.algo_version = None

    # default to outer environment
    # the difference are ossHost buckets arn settings
    self.is_outer = True

  def load_oss_config(self, config_path):
    with open(config_path, 'r') as fin:
      for line_str in fin:
        line_str = line_str.strip()
        line_str = line_str.replace(' ', '')
        if line_str.startswith('accessKeyID='):
          self.oss_key = line_str[len('accessKeyID='):].strip()
        elif line_str.startswith('accessKeySecret='):
          self.oss_secret = line_str[len('accessKeySecret='):].strip()
        elif line_str.startswith('endpoint='):
          self.endpoint = line_str[len('endpoint='):].strip()

  def load_odps_config(self, config_path):
    self.odps_config_path = config_path
    with open(config_path, 'r') as fin:
      for line_str in fin:
        line_str = line_str.strip()
        line_str = line_str.replace(' ', '')
        key_str = 'project_name='
        if line_str.startswith(key_str):
          self.project_name = line_str[len(key_str):]
        key_str = 'end_point='
        if line_str.startswith(key_str):
          self.odps_endpoint = line_str[len(key_str):]
        key_str = 'access_id='
        if line_str.startswith(key_str):
          self.dh_id = line_str[len(key_str):]
        key_str = 'access_key='
        if line_str.startswith(key_str):
          self.dh_key = line_str[len(key_str):]

  def clean_topic(self, dh_project):
    if not dh_project:
      return
    topic_names = self.dh.list_topic(dh_project).topic_names
    for topic_name in topic_names:
      self.clean_subscription(topic_name)
      self.dh.delete_topic(dh_project, topic_name)

  def clean_project(self):
    project_names = self.dh.list_project().project_names
    for dh_project in project_names:
      if dh_project == self.dh_project:
        self.clean_topic(dh_project)
        try:
          self.dh.delete_project(dh_project)
        except InvalidOperationException:
          pass

  def clean_subscription(self, topic_name):
    subscriptions = self.dh.list_subscription(self.dh_project, topic_name, '',
                                              1, 100).subscriptions
    for subscription in subscriptions:
      self.dh.delete_subscription(self.dh_project, topic_name, subscription)

  def get_input_type(self, input_type):
    DhDict = {
        'INT64': FieldType.BIGINT,
        'INT32': FieldType.BIGINT,
        'STRING': FieldType.STRING,
        'BOOLEAN': FieldType.BOOLEAN,
        'FLOAT32': FieldType.DOUBLE,
        'FLOAT64': FieldType.DOUBLE
    }

    return DhDict.get(input_type)

  def init_datahub(self):
    self.dh = DataHub(self.dh_id, self.dh_key, self.dh_endpoint)
    self.clean_project()
    try:
      self.dh.create_project(self.dh_project, comment='EasyRecTest')
      logging.info('create project success!')
    except ResourceExistException:
      logging.warning('project %s already exist!' % self.dh_project)
    except Exception:
      logging.error(traceback.format_exc())
    col_names = ['label', 'features']
    col_types = ['INT32', 'STRING']
    col_types = [self.get_input_type(x) for x in col_types]
    record_schema = RecordSchema.from_lists(col_names, col_types)
    try:
      if self.dh_topic in self.dh.list_topic(self.dh_project).topic_names:
        self.dh.delete_topic(self.dh_project, self.dh_topic)
      # project_name, topic_name, shard_count, life_cycle, record_schema, comment
      self.dh.create_tuple_topic(
          project_name=self.dh_project,
          topic_name=self.dh_topic,
          shard_count=2,
          life_cycle=1,
          record_schema=record_schema,
          comment='EasyRecTest')
      logging.info('create tuple topic %s succeed' % self.dh_topic)
    except ResourceExistException:
      logging.error('create topic %s failed: %s' %
                    (self.dh_topic, traceback.format_exc()))
    except Exception as ex:
      logging.error('exception: %s %s' % (str(ex), traceback.format_exc()))
    try:
      self.dh.wait_shards_ready(self.dh_project, self.dh_topic)
      logging.info('datahub[%s,%s] shards all ready' %
                   (self.dh_project, self.dh_topic))
      topic_result = self.dh.get_topic(self.dh_project, self.dh_topic)
      if topic_result.record_type != RecordType.TUPLE:
        logging.error('invalid topic type: %s' % str(topic_result.record_type))
      record_schema = topic_result.record_schema

      with open('data/test/dwd_avazu_ctr_deepmodel_10w.csv', 'r') as fin:
        record_list = []
        total_cnt = 0
        for line_str in fin:
          line_str = line_str.strip()
          sep_pos = line_str.find(',')
          label = int(line_str[:sep_pos])
          feature = line_str[(sep_pos + 1):]
          record = TupleRecord(values=[label, feature], schema=record_schema)
          record_list.append(record)
          if len(record_list) >= 8192:
            self.dh.put_records(self.dh_project, self.dh_topic, record_list)
            total_cnt += len(record_list)
            if total_cnt > 8192 * 64:
              break
    except Exception as ex:
      logging.error('exception: %s' % str(ex))
      logging.error(traceback.format_exc())


def get_oss_bucket(oss_key, oss_secret, endpoint, bucket_name):
  """Build oss2.Bucket instance.

  Args:
    oss_key: oss access_key
    oss_secret: oss access_secret
    endpoint: oss endpoint
    bucket_name: oss bucket name
  Return:
    oss2.Bucket instance
  """
  if oss_key is None or oss_secret is None:
    logging.info('oss_key or oss_secret is None')
    return None
  auth = oss2.Auth(oss_key, oss_secret)
  bucket = oss2.Bucket(auth, endpoint, bucket_name)
  return bucket


def delete_oss_path(bucket, in_prefix, bucket_name):
  """Delete oss path.

  Args:
    bucket: oss2.Bucket instance
    in_prefix: oss path prefix to be removed
    bucket_name: bucket_name
  """
  prefix = in_prefix.replace('oss://' + bucket_name + '/', '')
  for obj in oss2.ObjectIterator(bucket, prefix=prefix):
    bucket.delete_object(obj.key)
  bucket.delete_object(prefix)
  logging.info('delete oss path: %s, completed.' % in_prefix)
