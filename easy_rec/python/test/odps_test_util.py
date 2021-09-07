# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import configparser
import logging
import os
import time
import traceback

import oss2
from datahub import DataHub
# from datahub.exceptions import LimitExceededException
from datahub.exceptions import InvalidOperationException
from datahub.exceptions import ResourceExistException
# from datahub.exceptions import ResourceNotFoundException
# from datahub.models import BlobRecord
# from datahub.models import CursorType
from datahub.models import FieldType
from datahub.models import RecordSchema
from datahub.models import RecordType
from datahub.models import TupleRecord
from odps import ODPS
from odps.df import DataFrame


class OdpsOSSConfig:

  def __init__(self, script_path='./samples/odps_script'):
    self.time_stamp = int(time.time())
    temp_dir = os.environ.get('TEST_DIR', '/tmp')
    self.exp_dir = 'easy_rec_odps_test_%d' % self.time_stamp
    self.temp_dir = os.path.join(temp_dir, self.exp_dir)
    self.log_dir = os.path.join(self.temp_dir, 'logs/')

    # public buckets with readyonly access
    self.ali_bucket_endpoint = 'http://oss-cn-beijing.aliyuncs.com'
    self.ali_bucket_name = 'easyrec'
    self.script_path = script_path
    # read only access
    self.ali_oss_key = ''
    self.ali_oss_secret = ''

    self.oss_key = ''
    self.oss_secret = ''
    self.endpoint = ''
    self.arn = 'acs:ram::xxx:role/aliyunodpspaidefaultrole'
    self.bucket_name = ''

    self.odpscmd_path = os.environ.get('ODPS_CMD_PATH', 'odpscmd')
    self.odps_config_path = ''
    # input table project name replace {ODPS_PROJ_NAME} in
    # samples/odps_script:
    #    grep ODPS_PROJ_NAME -r samples/odps_script/
    self.project_name = ''

    self.dhid = ''
    self.dhkey = ''
    self.dhendpoint = ''
    self.dhtopic = ''
    self.dhproject = ''
    self.dh = ''
    self.odps = ''
    self.odpsEnd = ''

    # default to algo_public
    self.algo_project = None
    self.algo_res_project = None
    self.algo_version = None

  def load_dh_config(self, config_path):
    configer = configparser.ConfigParser()
    configer.read(config_path, encoding='utf-8')
    self.dhid = configer.get('datahub', 'access_id')
    self.dhkey = configer.get('datahub', 'access_key')
    self.dhendpoint = configer.get('datahub', 'endpoint')
    self.dhtopic = configer.get('datahub', 'topic_name')
    self.dhproject = configer.get('datahub', 'project')

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
        if line_str.startswith('project_name='):
          self.project_name = line_str[len('project_name='):]
        if line_str.startswith('end_point='):
          self.odpsEnd = line_str[len('end_point='):]

  def clean_topic(self, dhproject):
    if not dhproject:
      logging.error('project is empty .')
      topic_names = self.dh.list_topic(dhproject).topic_names
      for topic_name in topic_names:
        self.clean_subscription(topic_name)
        self.dh.delete_topic(dhproject, topic_name)

  def clean_project(self):
    project_names = self.dh.list_project().project_names
    for dhproject in project_names:
      if dhproject == self.dhproject:
        self.clean_topic(dhproject)
        try:
          self.dh.delete_project(dhproject)
        except InvalidOperationException:
          pass

  def clean_subscription(self, topic_name):
    subscriptions = self.dh.list_subscription(self.dhproject, topic_name, '', 1,
                                              100).subscriptions
    for subscription in subscriptions:
      self.dh.delete_subscription(self.dhproject, topic_name, subscription)

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

  def _subscription(self):
    self.dh = DataHub(self.dhid, self.dhkey, self.dhendpoint)
    self.odps = ODPS(self.dhid, self.dhkey, self.project_name, self.odpsEnd)
    self.odpsTable = 'deepfm_train_%s' % self.time_stamp
    self.clean_project()
    read_odps = DataFrame(self.odps.get_table(self.odpsTable))
    col = read_odps.schema.names
    col_type = [self.get_input_type(str(i)) for i in read_odps.schema.types]
    try:
      self.dh.create_project(self.dhproject, 'EasyRecTest')
      logging.info('create project success!')
    except ResourceExistException:
      logging.info('project %s already exist!' % self.dhproject)
    except Exception as ex:
      logging.info(traceback.format_exc(ex))
    record_schema = RecordSchema.from_lists(col, col_type)
    try:
      self.dh.create_tuple_topic(self.dhproject, self.dhtopic, 7, 3,
                                 record_schema, 'easyrec_datahub')
      logging.info('create tuple topic success!')
    except ResourceExistException:
      logging.info('topic %s already exist!' % self.dhtopic)
    except Exception as ex:
      logging.error('exception:', ex)
      logging.error(traceback.format_exc())
    try:
      self.dh.wait_shards_ready(self.dhproject, self.dhtopic)
      logging.info('shards all ready')
      topic_result = self.dh.get_topic(self.dhproject, self.dhtopic)
      if topic_result.record_type != RecordType.TUPLE:
        logging.error('topic type illegal! ')
      record_schema = topic_result.record_schema
      t = self.odps.get_table(self.odpsTable)
      with t.open_reader() as reader:
        size = 0
        record_list = []
        for data in reader[0:1000]:
          record = TupleRecord(values=data.values, schema=record_schema)
          record_list.append(record)
          if size % 1000:
            self.dh.put_records(self.dhproject, self.dhtopic, record_list)
            record_list = []
          size += 1

    except Exception as e:
      logging.error(e)


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
