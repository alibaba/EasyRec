import argparse
import logging
import os
import shutil
import sys

import tensorflow as tf

from easy_rec.python.test.odps_command import OdpsCommand
from easy_rec.python.test.odps_test_prepare import change_files
from easy_rec.python.test.odps_test_util import OdpsOSSConfig
from easy_rec.python.test.odps_test_util import delete_oss_path
from easy_rec.python.test.odps_test_util import get_oss_bucket
from easy_rec.python.utils import test_utils

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

DATAHUB_TEST_SCRIPT_PATH = './samples/dh_script'
odps_oss_config = OdpsOSSConfig(script_path=DATAHUB_TEST_SCRIPT_PATH)

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
    test_utils.test_datahub_train_eval(
        '%s/configs/deepfm.config' % odps_oss_config.temp_dir, odps_oss_config,
        self._test_dir, total_steps=10)
    self._success = False
    # self.assertTrue(self._success)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--odps_config', type=str, default=None, help='odps config path')
  parser.add_argument(
      '--oss_config', type=str, default=None, help='ossutilconfig path')
  parser.add_argument(
      '--bucket_name', type=str, default=None, help='test oss bucket name')
  # parser.add_argument('--arn', type=str, default=None, help='oss rolearn')
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

  assert args.odps_config is not None and args.odps_config != ''
  odps_oss_config.load_odps_config(args.odps_config)
  os.environ['ODPS_CONFIG_FILE_PATH'] = args.odps_config

  shutil.copytree(DATAHUB_TEST_SCRIPT_PATH, odps_oss_config.temp_dir)
  logging.info("temp_dir=%s" % odps_oss_config.temp_dir)
  for root, dirs, files in os.walk(odps_oss_config.temp_dir):
    for file in files:
      file_path = os.path.join(root, file)
      change_files(odps_oss_config, file_path)

  odps_oss_config.init_datahub()
  tf.test.main()

  # delete tmp
  shutil.rmtree(odps_oss_config.temp_dir)
