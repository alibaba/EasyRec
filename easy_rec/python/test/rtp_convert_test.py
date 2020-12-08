# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os

import tensorflow as tf

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


if __name__ == '__main__':
  tf.test.main()
