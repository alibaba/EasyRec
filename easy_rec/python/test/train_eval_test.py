# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import unittest
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from easy_rec.python.utils import test_utils


class TrainEvalTest(tf.test.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    self._test_dir = test_utils.get_tmp_dir()
    self._success = True
    logging.info('test dir: %s' % self._test_dir)

  def tearDown(self):
    test_utils.set_gpu_id(None)
    if self._success:
      test_utils.clean_up(self._test_dir)

  def test_deepfm_with_lookup_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_lookup.config', self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_combo_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_ctr.config', self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_vocab_list(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_vocab_list_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_multi_class(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_param_edit(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir,
        hyperparam_str='{"model_dir":"experiments/deepfm_multi_cls_on_avazu_ctr", '
        '"model_config.deepfm.wide_output_dim": 32}')
    self.assertTrue(self._success)

  def test_multi_tower(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_save_checkpoint_secs(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_save_secs_on_taobao.config',
        self._test_dir,
        total_steps=500)
    ckpts_times = []
    ckpt_dir = os.path.join(self._test_dir, 'train')
    for filepath in os.listdir(ckpt_dir):
      if filepath.startswith('model.ckpt') and filepath.endswith('meta'):
        ckpts_times.append(os.path.getmtime(os.path.join(ckpt_dir, filepath)))
    # remove last ckpt time
    ckpts_times = np.array(sorted(ckpts_times)[:-1])
    # ensure interval is 20s
    self.assertAllClose(
        ckpts_times[1:] - ckpts_times[:-1], [20] * (len(ckpts_times) - 1),
        atol=5)
    self.assertTrue(self._success)

  def test_multi_tower_with_best_exporter(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_best_export_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_fg_input(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg.config', self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_fg_json_config(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg.json', self._test_dir)
    self.assertTrue(self._success)

  def test_fm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_din(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/din_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_bst(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/bst_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dssm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dssm_with_regression(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_reg_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_regression(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_reg.config', self._test_dir)
    self.assertTrue(self._success)

  def test_mmoe(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mmoe_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_mmoe_deprecated(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mmoe_on_taobao_deprecated.config', self._test_dir)
    self.assertTrue(self._success)

  def test_simple_multi_task(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/simple_multi_task_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_essm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/esmm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_mmoe(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_mmoe_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_train_with_ps_worker(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(
      LooseVersion(tf.__version__) < LooseVersion('2.3.0'),
      'MultiWorkerMirroredStrategy need tf version > 2.3')
  def test_train_with_multi_worker_mirror(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_multi_worker_mirrored_strategy_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)


if __name__ == '__main__':
  tf.test.main()
