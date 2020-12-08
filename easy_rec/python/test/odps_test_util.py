# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import time

import oss2


class OdpsOSSConfig:

  def __init__(self):
    self.time_stamp = int(time.time())
    temp_dir = os.environ.get('TEST_DIR', '/tmp')
    self.exp_dir = 'easy_rec_odps_test_%d' % self.time_stamp
    self.temp_dir = os.path.join(temp_dir, self.exp_dir)
    self.log_dir = os.path.join(self.temp_dir, 'logs/')

    # public buckets with readyonly access
    self.ali_bucket_endpoint = 'http://oss-cn-beijing.aliyuncs.com'
    self.ali_bucket_name = 'easyrec'
    self.script_path = './samples/odps_script'
    # read only access
    self.ali_oss_key = 'LTAI4GHNdHdXvYQQW7NyHS8Y'
    self.ali_oss_secret = 'dOm1BrTGIVjmZwUnRFIapZreOD03Gw'

    self.oss_key = ''
    self.oss_secret = ''
    self.endpoint = ''
    self.arn = 'acs:ram::xxx:role/aliyunodpspaidefaultrole'
    self.bucket_name = ''

    self.odpscmd_path = 'odpscmd'
    self.odps_config_path = ''
    # input table project name replace {ODPS_PROJ_NAME} in
    # samples/odps_script:
    #    grep ODPS_PROJ_NAME -r samples/odps_script/
    self.project_name = ''

    # default to algo_public
    self.algo_project = None
    self.algo_res_project = None
    self.algo_version = None

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
