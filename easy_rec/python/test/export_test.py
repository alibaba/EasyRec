# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Date: 2020-10-06
# Filename：export_test.py
import json
import logging
import os

import tensorflow as tf

from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.utils import config_util
from easy_rec.python.utils import test_utils
from easy_rec.python.utils.test_utils import RunAsSubprocess


class ExportTest(tf.test.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def tearDown(self):
    test_utils.set_gpu_id(None)

  @RunAsSubprocess
  def _predict_and_check(self, data_path, saved_model_dir, cmp_result):
    predictor = Predictor(saved_model_dir)
    with open(data_path, 'r') as fin:
      inputs = []
      for line_str in fin:
        line_str = line_str.strip()
        line_tok = line_str.split(',')
        inputs.append(','.join(line_tok[1:]))
      output_res = predictor.predict(inputs, batch_size=32)

    for i in range(len(output_res)):
      prob0 = output_res[i]['probs']
      prob1 = cmp_result[i]['probs']
      self.assertAllClose(prob0, prob1, atol=1e-4)

  def test_export(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)

    # prepare model
    self.assertTrue(
        test_utils.test_single_train_eval(
            'samples/model_config/multi_tower_export.config',
            test_dir=test_dir))
    test_utils.set_gpu_id(None)

    # prepare two version config
    config_path_single = os.path.join(test_dir, 'pipeline.config')
    config_path_multi = os.path.join(test_dir, 'pipeline_v2.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        config_path_single)
    pipeline_config.export_config.multi_placeholder = False
    config_util.save_pipeline_config(pipeline_config, test_dir,
                                     'pipeline_v2.config')

    # prepare two version export dir
    export_dir_single = os.path.join(test_dir, 'train/export/final')
    export_dir_multi = os.path.join(test_dir, 'train/export/multi')
    export_cmd = """
      python -m easy_rec.python.export
        --pipeline_config_path %s
        --export_dir %s
    """ % (config_path_multi, export_dir_multi)
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_%s.txt' % (test_dir, 'export'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    # use checkpoint to prepare result
    result_path = os.path.join(test_dir, 'result.txt')
    predict_cmd = """
      python -m easy_rec.python.predict
        --pipeline_config_path %s
        --output_path %s
    """ % (config_path_single, result_path)
    proc = test_utils.run_cmd(predict_cmd % (),
                              '%s/log_%s.txt' % (test_dir, 'predict'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    with open(result_path, 'r') as fin:
      cmp_result = []
      for line_str in fin:
        line_str = line_str.strip()
        cmp_result.append(json.loads(line_str))

    test_data_path = 'data/test/export/data.csv'
    self._predict_and_check(test_data_path, export_dir_single, cmp_result)
    self._predict_and_check(test_data_path, export_dir_multi, cmp_result)
    test_utils.clean_up(test_dir)


if __name__ == '__main__':
  tf.test.main()
