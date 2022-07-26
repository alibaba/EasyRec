# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import json
import logging
import os
import platform
import unittest
from argparse import Namespace

if platform.python_version() >= '3.7':
  from easy_rec.python.hpo_nni.pai_nni.code.metric_utils import get_result
  from easy_rec.python.hpo_nni.pai_nni.code.utils import get_value
  from easy_rec.python.hpo_nni.pai_nni.code.utils import parse_config
  from easy_rec.python.hpo_nni.pai_nni.code.utils import set_value
  from easy_rec.python.hpo_nni.pai_nni.code.pyodps_utils import parse_easyrec_cmd_config

  from easy_rec.python.hpo_nni.pai_nni.code.modify_pipeline_config import get_learning_rate  # NOQA
  from easy_rec.python.hpo_nni.pai_nni.code.modify_pipeline_config import modify_config  # NOQA


class PAINNITest(unittest.TestCase):

  def __init__(self, methodName='HPOTest'):
    super(PAINNITest, self).__init__(methodName=methodName)
    filepath = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    self._metric_data_path = os.path.join(filepath,
                                          'data/test/hpo_test/eval_val/')
    self.config_path = os.path.join(
        filepath, 'easy_rec/python/hpo_nni/pai_nni/config/pipeline.config')
    self.save_path = os.path.join(
        filepath,
        'easy_rec/python/hpo_nni/pai_nni/config/pipeline_finetune.config')
    self.config_begin = os.path.join(
        filepath, 'easy_rec/python/hpo_nni/pai_nni/source_begin/config_begin')
    self.config_finetune = os.path.join(
        filepath,
        'easy_rec/python/hpo_nni/pai_nni/source_finetune/config_finetune')

  def test_get_metric(self):
    vals = get_result(None, self._metric_data_path)
    logging.info('eval result num = %d' % len(vals))
    logging.info('eval result[0] = %s' % json.dumps(vals[0]))

  def test_global_value(self):
    set_value('pai_nni', 1)
    y = get_value('pai_nni')
    assert y == 1

  def test_modify_pipeline(self):
    args = Namespace(
        pipeline_config_path=self.config_path,
        save_path=self.save_path,
        learning_rate=1e-6)
    modify_config(args)
    self.assertAlmostEqual(get_learning_rate(self.save_path), 1e-6)

  def test_parse_easyrec_config(self):
    config = parse_config(self.config_begin)

    metric_dict = ast.literal_eval(config['metric_hpo'])
    assert metric_dict['auc'] == 1

    command = parse_easyrec_cmd_config(config)
    assert command.name == 'easy_rec_ext'
    assert command.parameters['version'] == '0.4.2'

  def test_parse_easyrec_config_2(self):
    config = parse_config(self.config_finetune)

    metric_dict = ast.literal_eval(config['metric_hpo'])
    assert metric_dict['auc_is_valid_play'] == 0.5

    command = parse_easyrec_cmd_config(config)
    assert command.name == 'easy_rec_ext'
    assert command.parameters[
        'cluster'] == '{"ps":{"count":1,"cpu":1600,"memory":40000 },"worker":{"count":12,"cpu":1600,"memory":40000}}'


if __name__ == '__main__':
  if platform.python_version() >= '3.7':
    unittest.main()
