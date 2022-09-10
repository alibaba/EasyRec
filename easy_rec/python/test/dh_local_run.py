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

odps_oss_config = OdpsOSSConfig(script_path='./samples/dh_script')


class TestPipelineOnEmr(tf.test.TestCase):
  """Train eval test on emr."""

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    self._success = True
    self._test_dir = test_utils.get_tmp_dir()
    logging.info('test datahub local dir: %s' % self._test_dir)

  def tearDown(self):
    if self._success:
      shutil.rmtree(self._test_dir)

  def test_datahub_train_eval(self):
    end = ['deep_fm/drop_table.sql']
    odps_cmd = OdpsCommand(odps_oss_config)

    self._success = test_utils.test_datahub_train_eval(
        '%s/configs/deepfm.config' % odps_oss_config.temp_dir, odps_oss_config,
        self._test_dir)
    odps_cmd.run_list(end)
    self.assertTrue(self._success)


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
  prepare(odps_oss_config)
  start = [
      'deep_fm/create_external_deepfm_table.sql',
      'deep_fm/create_inner_deepfm_table.sql'
  ]
  end = ['deep_fm/drop_table.sql']
  odps_cmd = OdpsCommand(odps_oss_config)
  odps_cmd.run_list(start)
  odps_oss_config.init_dh_and_odps()
  tf.test.main()
  # delete oss path
  bucket = get_oss_bucket(odps_oss_config.oss_key, odps_oss_config.oss_secret,
                          odps_oss_config.endpoint, odps_oss_config.bucket_name)
  delete_oss_path(bucket, odps_oss_config.exp_dir, odps_oss_config.bucket_name)
  # delete tmp
  shutil.rmtree(odps_oss_config.temp_dir)
