# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os

import tensorflow as tf

from easy_rec.python.utils import test_utils


class ExcelConvertTest(tf.test.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def tearDown(self):
    test_utils.set_gpu_id(None)

  def test_deepfm_convert(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)
    pipeline_config_path = os.path.join(test_dir, 'avazu_deepfm_excel.config')
    convert_cmd = """
      python -m easy_rec.python.tools.create_config_from_excel
        --excel_path samples/excel_config/dwd_avazu_ctr_deepfm.xls
        --model_type deepfm
        --output_path  %s
        --train_input_path data/test/dwd_avazu_ctr_deepmodel_10w.csv
        --eval_input_path data/test/dwd_avazu_ctr_deepmodel_10w.csv
    """ % pipeline_config_path
    proc = test_utils.run_cmd(convert_cmd,
                              '%s/log_%s.txt' % (test_dir, 'convert'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.clean_up(test_dir)

  def test_multi_tower_convert(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)
    pipeline_config_path = os.path.join(test_dir, 'avazu_deepfm_excel.config')
    convert_cmd = """
      python -m easy_rec.python.tools.create_config_from_excel
        --excel_path samples/excel_config/dwd_avazu_ctr_deepfm.xls
        --model_type multi_tower
        --output_path  %s
        --train_input_path data/test/dwd_avazu_ctr_deepmodel_10w.csv
        --eval_input_path data/test/dwd_avazu_ctr_deepmodel_10w.csv
    """ % pipeline_config_path
    proc = test_utils.run_cmd(convert_cmd,
                              '%s/log_%s.txt' % (test_dir, 'convert'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.clean_up(test_dir)


if __name__ == '__main__':
  tf.test.main()
