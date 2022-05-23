# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging

import tensorflow as tf

from easy_rec.python.utils import test_utils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
gfile = tf.gfile


class CheckTest(tf.test.TestCase):

  def setUp(self):
    self._test_dir = test_utils.get_tmp_dir()
    self._success = True
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    logging.info('test dir: %s' % self._test_dir)

  def tearDown(self):
    test_utils.set_gpu_id(None)
    if self._success:
      test_utils.clean_up(self._test_dir)

  def test_csv_input_train_with_check(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_taobao.config',
        self._test_dir,
        check_mode=True)
    self.assertTrue(self._success)

  def test_rtp_input_train_with_check(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg.config',
        self._test_dir,
        check_mode=True)
    self.assertTrue(self._success)

  def test_csv_input_with_pre_check(self):
    self._success = test_utils.test_single_pre_check(
        'samples/model_config/dbmtl_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_rtp_input_with_pre_check(self):
    self._success = test_utils.test_single_pre_check(
        'samples/model_config/dbmtl_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)


if __name__ == '__main__':
  tf.test.main()
