# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import unittest

import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.utils import config_util
from easy_rec.python.utils import fg_util
from easy_rec.python.utils import test_utils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class FGTest(tf.test.TestCase):

  def __init__(self, methodName='FGTest'):
    super(FGTest, self).__init__(methodName=methodName)

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    self._test_dir = test_utils.get_tmp_dir()
    self._success = True
    logging.info('test dir: %s' % self._test_dir)

  def tearDown(self):
    test_utils.set_gpu_id(None)
    if self._success:
      test_utils.clean_up(self._test_dir)

  def test_fg_json_to_config(self):
    pipeline_config_path = 'samples/rtp_fg/fg_test_extensions.config'
    final_pipeline_config_path = 'samples/rtp_fg/fg_test_extensions_final.config'
    fg_path = 'samples/rtp_fg/fg_test_extensions.json'

    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path)
    pipeline_config.fg_json_path = fg_path
    fg_util.load_fg_json_to_config(pipeline_config)
    pipeline_config_str = text_format.MessageToString(
        pipeline_config, as_utf8=True)

    final_pipeline_config = config_util.get_configs_from_pipeline_file(
        final_pipeline_config_path)
    final_pipeline_config_str = text_format.MessageToString(
        final_pipeline_config, as_utf8=True)
    self.assertEqual(pipeline_config_str, final_pipeline_config_str)

  def test_fg_dtype(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg_test_dtype.config', self._test_dir)
    self.assertTrue(self._success)

  def test_fg_train(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fg_train.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf('-PAI' not in tf.__version__,
                   'Only test when pai-tf is used.')
  def test_fg_train_ev(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fg_train_ev.config', self._test_dir)
    self.assertTrue(self._success)


if __name__ == '__main__':
  tf.test.main()
