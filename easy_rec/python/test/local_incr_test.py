# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.utils import numpy_utils
from easy_rec.python.utils import test_utils


class LocalIncrTest(tf.test.TestCase):

  def setUp(self):
    self._success = True
    self._test_dir = test_utils.get_tmp_dir()

    logging.info('Testing %s.%s, test_dir=%s' %
                 (type(self).__name__, self._testMethodName, self._test_dir))
    self._log_dir = os.path.join(self._test_dir, 'logs')
    if not gfile.IsDirectory(self._log_dir):
      gfile.MakeDirs(self._log_dir)

  @unittest.skipIf(
      'oss_path' not in os.environ or
      'oss_endpoint' not in os.environ and 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ, 'Only execute when kafka is available')
  def test_incr_save(self):
    self._test_incr_save(
        'samples/model_config/taobao_fg_incr_save_local.config')

  @unittest.skipIf(
      'oss_path' not in os.environ or
      'oss_endpoint' not in os.environ and 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ, 'Only execute when kafka is available')
  def test_incr_save_ev(self):
    self._test_incr_save(
        'samples/model_config/taobao_fg_incr_save_ev_local.config')

  @unittest.skipIf(
      'oss_path' not in os.environ or
      'oss_endpoint' not in os.environ and 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ, 'Only execute when kafka is available')
  def test_incr_save_share_ev(self):
    self._test_incr_save(
        'samples/model_config/taobao_fg_incr_save_share_ev_local.config')

  def _test_incr_save(self, config_path):
    self._success = False
    success = test_utils.test_distributed_train_eval(
        config_path,
        self._test_dir,
        total_steps=100,
        edit_config_json={
            'train_config.incr_save_config.fs.mount_path':
                os.path.join(self._test_dir, 'train/incr_save/')
        })
    self.assertTrue(success)
    export_cmd = """
       python -m easy_rec.python.export --pipeline_config_path %s/pipeline.config
           --export_dir %s/export/sep/ --oss_path=%s --oss_ak=%s --oss_sk=%s --oss_endpoint=%s
           --asset_files ./samples/rtp_fg/fg.json
           --checkpoint_path %s/train/model.ckpt-0
    """ % (self._test_dir, self._test_dir, os.environ['oss_path'],
           os.environ['oss_ak'], os.environ['oss_sk'],
           os.environ['oss_endpoint'], self._test_dir)
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_export_sep.txt' % self._test_dir)
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    files = gfile.Glob(os.path.join(self._test_dir, 'export/sep/[1-9][0-9]*'))
    export_sep_dir = files[0]

    predict_cmd = """
        python -m easy_rec.python.inference.processor.test --saved_model_dir %s
           --input_path data/test/rtp/taobao_test_feature.txt
           --output_path %s/processor.out  --test_dir %s
     """ % (export_sep_dir, self._test_dir, self._test_dir)
    envs = dict(os.environ)
    envs['PROCESSOR_TEST'] = '1'
    proc = test_utils.run_cmd(
        predict_cmd, '%s/log_processor.txt' % self._test_dir, env=envs)
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    with open('%s/processor.out' % self._test_dir, 'r') as fin:
      processor_out = []
      for line_str in fin:
        line_str = line_str.strip()
        processor_out.append(json.loads(line_str))

    predictor = Predictor(os.path.join(self._test_dir, 'train/export/final/'))
    with open('data/test/rtp/taobao_test_feature.txt', 'r') as fin:
      inputs = []
      for line_str in fin:
        line_str = line_str.strip()
        line_tok = line_str.split(';')[-1]
        line_tok = line_tok.split(chr(2))
        inputs.append(line_tok)
    output_res = predictor.predict(inputs, batch_size=1024)

    with open('%s/predictor.out' % self._test_dir, 'w') as fout:
      for i in range(len(output_res)):
        fout.write(
            json.dumps(output_res[i], cls=numpy_utils.NumpyEncoder) + '\n')

    for i in range(len(output_res)):
      val0 = output_res[i]['probs']
      val1 = processor_out[i]['probs']
      diff = np.abs(val0 - val1)
      assert diff < 1e-4, 'too much difference[%.6f] >= 1e-4' % diff
    self._success = True


if __name__ == '__main__':
  tf.test.main()
