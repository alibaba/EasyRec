import argparse
import logging
import os
import shutil
import sys

import tensorflow as tf

from easy_rec.python.test.odps_test_prepare import change_files
from easy_rec.python.test.odps_test_util import OdpsOSSConfig
from easy_rec.python.utils import config_util
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

  def _load_config_for_test(self, config_path, total_steps=50):
    pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
    pipeline_config.train_config.train_distribute = 0
    pipeline_config.train_config.sync_replicas = False

    pipeline_config.datahub_train_input.akId = odps_oss_config.dh_id
    pipeline_config.datahub_train_input.akSecret = odps_oss_config.dh_key
    pipeline_config.datahub_train_input.endpoint = odps_oss_config.dh_endpoint
    pipeline_config.datahub_train_input.project = odps_oss_config.dh_project
    pipeline_config.datahub_train_input.topic = odps_oss_config.dh_topic

    pipeline_config.datahub_eval_input.akId = odps_oss_config.dh_id
    pipeline_config.datahub_eval_input.akSecret = odps_oss_config.dh_key
    pipeline_config.datahub_eval_input.endpoint = odps_oss_config.dh_endpoint
    pipeline_config.datahub_eval_input.project = odps_oss_config.dh_project
    pipeline_config.datahub_eval_input.topic = odps_oss_config.dh_topic
    return pipeline_config

  def test_datahub_train_eval(self):
    config_path = 'samples/dh_script/configs/deepfm.config'
    pipeline_config = self._load_config_for_test(config_path)
    test_utils.test_single_train_eval(
        pipeline_config, self._test_dir, total_steps=10)
    self.assertTrue(self._success)

  def test_distributed_datahub_train_eval(self):
    config_path = 'samples/dh_script/configs/deepfm.config'
    pipeline_config = self._load_config_for_test(config_path)
    pipeline_config.data_config.chief_redundant = True
    test_utils.test_distributed_train_eval(
        pipeline_config, self._test_dir, total_steps=10)
    self.assertTrue(self._success)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--odps_config', type=str, default=None, help='odps config path')
  args, unknown_args = parser.parse_known_args()

  sys.argv = [sys.argv[0]]
  for unk_arg in unknown_args:
    sys.argv.append(unk_arg)

  assert args.odps_config is not None and args.odps_config != ''
  odps_oss_config.load_odps_config(args.odps_config)
  os.environ['ODPS_CONFIG_FILE_PATH'] = args.odps_config

  shutil.copytree(DATAHUB_TEST_SCRIPT_PATH, odps_oss_config.temp_dir)
  logging.info('temp_dir=%s' % odps_oss_config.temp_dir)
  for root, dirs, files in os.walk(odps_oss_config.temp_dir):
    for file in files:
      file_path = os.path.join(root, file)
      change_files(odps_oss_config, file_path)

  odps_oss_config.init_datahub()
  tf.test.main()

  # delete tmp
  shutil.rmtree(odps_oss_config.temp_dir)
