# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import os
import shutil
import sys

import tensorflow as tf

from easy_rec.python.test.odps_command import OdpsCommand
from easy_rec.python.test.odps_test_prepare import prepare
from easy_rec.python.test.odps_test_util import OdpsOSSConfig
from easy_rec.python.test.odps_test_util import delete_oss_path
from easy_rec.python.test.odps_test_util import get_oss_bucket
from easy_rec.python.utils import test_utils

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

odps_oss_config = OdpsOSSConfig(script_path='./samples/emr_script')


class TestPipelineOnEmr(tf.test.TestCase):
  """Train eval test on emr."""

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    self._test_hdfs_dir = test_utils.get_hdfs_tmp_dir(
        'hdfs://emr-header-1:9000/user/easy_rec/emr_test')
    self._success = True
    logging.info('test hdfs dir: %s' % self._test_hdfs_dir)

  def tearDown(self):
    if self._success:
      pass
    test_utils.clean_up_hdfs(self._test_hdfs_dir)

  def test_deepfm_train_eval_export(self):
    start = [
        'deep_fm/create_external_deepfm_table.sql',
        'deep_fm/create_inner_deepfm_table.sql'
    ]
    end = ['deep_fm/drop_table.sql']
    odps_cmd = OdpsCommand(odps_oss_config)
    odps_cmd.run_list(start)
    self._success = test_utils.test_hdfs_train_eval(
        '%s/configs/deepfm.config' % odps_oss_config.temp_dir,
        '%s/yaml_config/train.paitf.yaml' % odps_oss_config.temp_dir,
        self._test_hdfs_dir)
    self.assertTrue(self._success)

    self._success = test_utils.test_hdfs_eval(
        '%s/configs/deepfm_eval_pipeline.config' % odps_oss_config.temp_dir,
        '%s/yaml_config/eval.tf.yaml' % odps_oss_config.temp_dir,
        self._test_hdfs_dir)
    self.assertTrue(self._success)

    self._success = test_utils.test_hdfs_export(
        '%s/configs/deepfm_eval_pipeline.config' % odps_oss_config.temp_dir,
        '%s/yaml_config/export.tf.yaml' % odps_oss_config.temp_dir,
        self._test_hdfs_dir)
    self.assertTrue(self._success)

    odps_cmd.run_list(end)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--odps_config', type=str, default=None, help='odps config path')
  parser.add_argument(
      '--oss_config', type=str, default=None, help='ossutilconfig path')
  parser.add_argument(
      '--bucket_name', type=str, default=None, help='test oss bucket name')
  parser.add_argument('--arn', type=str, default=None, help='oss rolearn')
  parser.add_argument(
      '--odpscmd', type=str, default='odpscmd', help='odpscmd path')
  parser.add_argument(
      '--algo_project', type=str, default=None, help='algo project name')
  parser.add_argument(
      '--algo_res_project',
      type=str,
      default=None,
      help='algo resource project name')
  parser.add_argument(
      '--algo_version', type=str, default=None, help='algo version')
  args, unknown_args = parser.parse_known_args()
  sys.argv = [sys.argv[0]]
  for unk_arg in unknown_args:
    sys.argv.append(unk_arg)

  if args.odps_config:
    odps_oss_config.load_odps_config(args.odps_config)
    os.environ['ODPS_CONFIG_FILE_PATH'] = args.odps_config
  if args.oss_config:
    odps_oss_config.load_oss_config(args.oss_config)
  if args.odpscmd:
    odps_oss_config.odpscmd_path = args.odpscmd
  if args.algo_project:
    odps_oss_config.algo_project = args.algo_project
  if args.algo_res_project:
    odps_oss_config.algo_res_project = args.algo_res_project
  if args.algo_version:
    odps_oss_config.algo_version = args.algo_version
  if args.arn:
    odps_oss_config.arn = args.arn
  if args.bucket_name:
    odps_oss_config.bucket_name = args.bucket_name
  print(args)
  prepare(odps_oss_config)
  tf.test.main()
  # delete oss path
  bucket = get_oss_bucket(odps_oss_config.oss_key, odps_oss_config.oss_secret,
                          odps_oss_config.endpoint, odps_oss_config.bucket_name)
  delete_oss_path(bucket, odps_oss_config.exp_dir, odps_oss_config.bucket_name)
  # delete tmp
  shutil.rmtree(odps_oss_config.temp_dir)
