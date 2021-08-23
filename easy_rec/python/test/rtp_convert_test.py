# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os

import tensorflow as tf

from easy_rec.python.utils import config_util
from easy_rec.python.utils import test_utils


class RTPConvertTest(tf.test.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def tearDown(self):
    test_utils.set_gpu_id(None)

  def test_rtp_convert(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)
    pipeline_config_path = os.path.join(test_dir, 'fg_multi_tower.config')
    convert_cmd = """
      python -m easy_rec.python.tools.convert_rtp_fg
        --rtp_fg samples/rtp_fg/fg.json
        --label clk
        --output_path  %s
        --input_type RTPInput
        --model_type multi_tower
        --train_input_path data/test/rtp/taobao_train_feature.txt
        --eval_input_path data/test/rtp/taobao_test_feature.txt
        --selected_cols 0,3 --num_steps 400
    """ % pipeline_config_path
    proc = test_utils.run_cmd(convert_cmd,
                              '%s/log_%s.txt' % (test_dir, 'convert'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.clean_up(test_dir)

  def test_rtp_convert_bucketize(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)
    pipeline_config_path = os.path.join(test_dir, 'fg_multi_tower.config')
    convert_cmd = """
      python -m easy_rec.python.tools.convert_rtp_fg
        --rtp_fg samples/rtp_fg/fg_bucketize.json
        --label clk
        --output_path  %s
        --input_type RTPInput
        --model_type multi_tower
        --train_input_path data/test/rtp/taobao_train_bucketize_feature.txt
        --eval_input_path data/test/rtp/taobao_test_bucketize_feature.txt
        --selected_cols 0,3 --num_steps 400
    """ % pipeline_config_path
    proc = test_utils.run_cmd(convert_cmd,
                              '%s/log_%s.txt' % (test_dir, 'convert'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.clean_up(test_dir)

  def test_rtp_convert_bucketize_v2(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)
    pipeline_config_path = os.path.join(test_dir, 'fg_multi_tower.config')
    convert_cmd = """
      python -m easy_rec.python.tools.convert_rtp_fg
        --rtp_fg samples/rtp_fg/fg_bucketize_v2.json
        --label clk
        --output_path  %s
        --input_type RTPInput
        --model_type multi_tower
        --train_input_path data/test/rtp/taobao_train_feature.txt
        --eval_input_path data/test/rtp/taobao_test_feature.txt
        --selected_cols 0,3 --num_steps 400
    """ % pipeline_config_path
    proc = test_utils.run_cmd(convert_cmd,
                              '%s/log_%s.txt' % (test_dir, 'convert'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    tmp_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path)
    for feature_config in tmp_config.feature_configs:
      if feature_config.input_names[0] == 'price':
        assert len(feature_config.boundaries) == 6

    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.clean_up(test_dir)

  def test_rtp_convert_test_model_config(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)
    pipeline_config_path = os.path.join(test_dir, 'fg_wide_and_deep.config')
    convert_cmd = """
      python -m easy_rec.python.tools.convert_rtp_fg
        --rtp_fg samples/rtp_fg/fg_bucketize_model_config.json
        --label clk
        --output_path  %s
        --input_type RTPInput
        --train_input_path data/test/rtp/taobao_train_feature.txt
        --eval_input_path data/test/rtp/taobao_test_feature.txt
        --selected_cols 0,3 --num_steps 400
    """ % pipeline_config_path
    proc = test_utils.run_cmd(convert_cmd,
                              '%s/log_%s.txt' % (test_dir, 'convert'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    tmp_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path)
    assert len(tmp_config.model_config.wide_and_deep.dnn.hidden_units) == 2
    assert tmp_config.model_config.wide_and_deep.dnn.hidden_units[0] == 48
    assert tmp_config.model_config.wide_and_deep.dnn.hidden_units[1] == 24
    assert tmp_config.model_dir == 'experiments/rtp_fg/wide_and_deep_update_model'

    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.clean_up(test_dir)


if __name__ == '__main__':
  tf.test.main()
