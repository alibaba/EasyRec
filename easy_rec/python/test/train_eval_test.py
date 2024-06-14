# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import logging
import os
import threading
import time
import unittest
from distutils.version import LooseVersion

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.main import predict
from easy_rec.python.utils import config_util
from easy_rec.python.utils import constant
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import test_utils

try:
  import graphlearn as gl
except Exception:
  gl = None

try:
  import horovod as hvd
except Exception:
  hvd = None

try:
  from sparse_operation_kit import experiment as sok
except Exception:
  sok = None

tf_version = tf.__version__
if tf.__version__ >= '2.0':
  tf = tf.compat.v1


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

  def test_deepfm_with_combo_v2_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_v2_on_avazu_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_combo_v3_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_v3_on_avazu_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_freeze_gradient(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_freeze_gradient.config', self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_vocab_list(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_vocab_list_on_avazu_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_multi_class(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_wide_and_deep_no_final(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/wide_and_deep_no_final_on_avazau_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_wide_and_deep(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/wide_and_deep_on_avazau_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_wide_and_deep_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/wide_and_deep_backbone_on_avazau.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dlrm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dlrm_on_taobao.config', self._test_dir)

  def test_dlrm_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dlrm_backbone_on_taobao.config', self._test_dir)

  def test_adamw_optimizer(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_adamw_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_momentumw_optimizer(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_momentumw_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_param_edit(self):
    model_dir = os.path.join(self._test_dir, 'train_new')
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir,
        hyperparam_str='{"model_dir":"%s", '
        '"model_config.deepfm.wide_output_dim": 32}' % model_dir)
    self.assertTrue(self._success)
    config_path = os.path.join(model_dir, 'pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
    self.assertTrue(pipeline_config.model_dir == model_dir)
    self.assertTrue(pipeline_config.model_config.deepfm.wide_output_dim == 32)

  def test_multi_tower(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_backbone_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_gauc(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao_gauc.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_session_auc(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao_session_auc.config',
        self._test_dir)
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
    diffs = list(ckpts_times[1:] - ckpts_times[:-1])
    logging.info('nearby ckpts_times diff = %s' % diffs)
    self.assertAllClose(
        ckpts_times[1:] - ckpts_times[:-1], [20] * (len(ckpts_times) - 1),
        atol=20)
    self.assertTrue(self._success)

  def test_keep_ckpt_max(self):

    def _post_check_func(pipeline_config):
      ckpt_prefix = os.path.join(pipeline_config.model_dir, 'model.ckpt-*.meta')
      ckpts = gfile.Glob(ckpt_prefix)
      assert len(ckpts) == 3, 'invalid number of checkpoints: %d' % len(ckpts)

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_ckpt_keep_3_on_taobao.config',
        self._test_dir,
        total_steps=500,
        post_check_func=_post_check_func)

  def test_multi_tower_with_best_exporter(self):

    def _post_check_func(pipeline_config):
      model_dir = pipeline_config.model_dir
      best_ckpts = os.path.join(model_dir, 'best_ckpt/model.ckpt-*.meta')
      best_ckpts = gfile.Glob(best_ckpts)
      assert len(best_ckpts) <= 2, 'too many best ckpts: %s' % str(best_ckpts)
      best_exports = os.path.join(model_dir, 'export/best/*')
      best_exports = gfile.Glob(best_exports)
      assert len(
          best_exports) <= 2, 'too many best exports: %s' % str(best_exports)
      return True

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_best_export_on_taobao.config',
        self._test_dir,
        total_steps=800,
        post_check_func=_post_check_func,
        timeout=3000)
    self.assertTrue(self._success)

  def test_latest_ckpt(self):
    tmp = estimator_utils.latest_checkpoint('data/test/latest_ckpt_test')
    assert tmp.endswith('model.ckpt-500')
    tmp = estimator_utils.latest_checkpoint('data/test/latest_ckpt_test/')
    assert tmp.endswith('model.ckpt-500')

  def test_latest_ckpt_v2(self):

    def _post_check_func(pipeline_config):
      logging.info('model_dir: %s' % pipeline_config.model_dir)
      logging.info('latest_checkpoint: %s' %
                   estimator_utils.latest_checkpoint(pipeline_config.model_dir))
      return tf.train.latest_checkpoint(pipeline_config.model_dir) == \
          estimator_utils.latest_checkpoint(pipeline_config.model_dir)

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg.config',
        self._test_dir,
        post_check_func=_post_check_func)
    self.assertTrue(self._success)

  def test_oss_stop_signal(self):
    train_dir = os.path.join(self._test_dir, 'train/')

    def _watch_func():
      while True:
        tmp_ckpt = estimator_utils.latest_checkpoint(train_dir)
        if tmp_ckpt is not None:
          version = estimator_utils.get_ckpt_version(tmp_ckpt)
          if version > 30:
            break
        time.sleep(1)
      stop_file = os.path.join(train_dir, 'OSS_STOP_SIGNAL')
      with open(stop_file, 'w') as fout:
        fout.write('OSS_STOP_SIGNAL')

    watch_th = threading.Thread(target=_watch_func)
    watch_th.start()

    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/taobao_fg_signal_stop.config',
        self._test_dir,
        total_steps=1000)
    self.assertTrue(self._success)
    watch_th.join()
    final_ckpt = estimator_utils.latest_checkpoint(train_dir)
    ckpt_version = estimator_utils.get_ckpt_version(final_ckpt)
    logging.info('final ckpt version = %d' % ckpt_version)
    self._success = ckpt_version < 1000
    assert ckpt_version < 1000

  def test_dead_line_stop_signal(self):
    train_dir = os.path.join(self._test_dir, 'train/')
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/dead_line_stop.config',
        self._test_dir,
        total_steps=1000)
    self.assertTrue(self._success)
    final_ckpt = estimator_utils.latest_checkpoint(train_dir)
    ckpt_version = estimator_utils.get_ckpt_version(final_ckpt)
    logging.info('final ckpt version = %d' % ckpt_version)
    self._success = ckpt_version < 1000
    assert ckpt_version < 1000

  def test_fine_tune_latest_ckpt_path(self):

    def _post_check_func(pipeline_config):
      logging.info('model_dir: %s' % pipeline_config.model_dir)
      pipeline_config = config_util.get_configs_from_pipeline_file(
          os.path.join(pipeline_config.model_dir, 'pipeline.config'), False)
      logging.info('fine_tune_checkpoint: %s' %
                   pipeline_config.train_config.fine_tune_checkpoint)
      return pipeline_config.train_config.fine_tune_checkpoint == \
          'data/test/mt_ckpt/model.ckpt-100'

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao.config',
        self._test_dir,
        fine_tune_checkpoint='data/test/mt_ckpt',
        post_check_func=_post_check_func)
    self.assertTrue(self._success)

  def test_fine_tune_ckpt(self):

    def _post_check_func(pipeline_config):
      pipeline_config.train_config.fine_tune_checkpoint = \
          estimator_utils.latest_checkpoint(pipeline_config.model_dir)
      test_dir = os.path.join(self._test_dir, 'fine_tune')
      pipeline_config.model_dir = os.path.join(test_dir, 'ckpt')
      return test_utils.test_single_train_eval(pipeline_config, test_dir)

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg.config',
        self._test_dir,
        post_check_func=_post_check_func)
    self.assertTrue(self._success)

  def test_multi_tower_multi_value_export(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_multi_value_export_on_taobao.config',
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

  def test_place_embed_on_cpu(self):
    os.environ['place_embedding_on_cpu'] = 'True'
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_din(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/din_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_din_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/din_backbone_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_bst(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/bst_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_bst_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/bst_backbone_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_cl4srec(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cl4srec_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dcn(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dcn_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_fibinet(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fibinet_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_masknet(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/masknet_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dcn_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dcn_backbone_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dcn_with_f1(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dcn_f1_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_autoint(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/autoint_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_uniter(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/uniter_on_movielens.config', self._test_dir)
    self.assertTrue(self._success)

  def test_highway(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/highway_on_movielens.config', self._test_dir)
    self.assertTrue(self._success)

  def test_cdn(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cdn_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_ppnet(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/ppnet_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_uniter_only_text_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/uniter_on_movielens_only_text_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_uniter_only_image_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/uniter_on_movielens_only_image_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_cmbf(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cmbf_on_movielens.config', self._test_dir)
    self.assertTrue(self._success)

  def test_cmbf_with_multi_loss(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cmbf_with_multi_loss.config', self._test_dir)
    self.assertTrue(self._success)

  def test_cmbf_has_other_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cmbf_on_movielens_has_other_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_cmbf_only_text_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cmbf_on_movielens_only_text_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_cmbf_only_image_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/cmbf_on_movielens_only_image_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dssm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dropoutnet(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dropoutnet_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_metric_learning(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/metric_learning_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_neg_sampler(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_neg_sampler_v2(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_v2_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_hard_neg_sampler(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_hard_neg_sampler_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_hard_neg_regular_sampler(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_hard_neg_sampler_regular_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_hard_neg_sampler_v2(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_hard_neg_sampler_v2_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dssm_no_norm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_inner_prod_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  # def test_dssm_with_regression(self):
  #   self._success = test_utils.test_single_train_eval(
  #       'samples/model_config/dssm_reg_on_taobao.config', self._test_dir)
  #   self.assertTrue(self._success)

  def _test_kd(self, config0, config1):
    self._success = test_utils.test_single_train_eval(config0, self._test_dir)
    self.assertTrue(self._success)
    config_path = os.path.join(self._test_dir, 'pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(config_path)

    train_path = os.path.join(self._test_dir, 'train_kd')
    eval_path = os.path.join(self._test_dir, 'eval_kd')

    @test_utils.RunAsSubprocess
    def _gen_kd_data(train_path, eval_path):
      pred_result = predict(config_path, None, pipeline_config.train_input_path)
      with gfile.GFile(pipeline_config.train_input_path, 'r') as fin:
        with gfile.GFile(train_path, 'w') as fout:
          for line, pred in zip(fin, pred_result):
            if isinstance(pred['logits'], type(np.array([]))):
              pred_logits = ''.join([str(x) for x in pred['logits']])
            else:
              pred_logits = str(pred['logits'])
            fout.write(line.strip() + ',' + pred_logits + '\n')
      pred_result = predict(config_path, None, pipeline_config.eval_input_path)
      with gfile.GFile(pipeline_config.eval_input_path, 'r') as fin:
        with gfile.GFile(eval_path, 'w') as fout:
          for line, pred in zip(fin, pred_result):
            if isinstance(pred['logits'], type(np.array([]))):
              pred_logits = ''.join([str(x) for x in pred['logits']])
            else:
              pred_logits = str(pred['logits'])
            fout.write(line.strip() + ',' + pred_logits + '\n')

    _gen_kd_data(train_path, eval_path)
    pipeline_config = config_util.get_configs_from_pipeline_file(config1)
    pipeline_config.train_input_path = train_path
    pipeline_config.eval_input_path = eval_path
    config_util.save_pipeline_config(pipeline_config, self._test_dir,
                                     'kd_pipeline.config')
    self._success = test_utils.test_single_train_eval(
        os.path.join(self._test_dir, 'kd_pipeline.config'),
        os.path.join(self._test_dir, 'kd'))
    self.assertTrue(self._success)

  def test_dssm_with_kd(self):
    self._test_kd('samples/model_config/multi_tower_on_taobao.config',
                  'samples/model_config/dssm_kd_on_taobao.config')

  def test_deepfm_multi_class_with_kd(self):
    self._test_kd('samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
                  'samples/model_config/deepfm_multi_cls_small.config')

  def test_mind(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mind_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_mind_with_time_id(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mind_on_taobao_with_time.config', self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_regression(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_reg.config', self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_sigmoid_l2_loss(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_sigmoid_l2.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_embedding_learning_rate(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_emblr_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_eval_online(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_eval_online_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_with_eval_online_gauc(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_eval_online_gauc_ctr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_mmoe(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mmoe_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_mmoe_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mmoe_backbone_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_mmoe_with_multi_loss(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mmoe_on_taobao_with_multi_loss.config',
        self._test_dir)
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

  def test_simple_multi_task_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/simple_multi_task_backbone_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_esmm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/esmm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_tag_kv_input(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/kv_tag.config', self._test_dir)
    self.assertTrue(self._success)

  def test_aitm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/aitm_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_backbone(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_backbone_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_cmbf(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_cmbf_on_movielens.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_uniter(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_uniter_on_movielens.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_with_multi_loss(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_taobao_with_multi_loss.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_early_stop(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_early_stop_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_early_stop_custom(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/custom_early_stop_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_early_stop_dis(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_early_stop_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_latest_export_with_asset(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/din_on_taobao_latest_export.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_incompatible_restore(self):

    def _post_check_func(config):
      config.feature_config.features[0].hash_bucket_size += 20000
      config.feature_config.features[1].hash_bucket_size += 100
      config.train_config.fine_tune_checkpoint = config.model_dir
      config.model_dir += '_finetune'
      config.train_config.force_restore_shape_compatible = True
      return test_utils.test_single_train_eval(
          config, os.path.join(self._test_dir, 'finetune'))

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg.config',
        self._test_dir,
        post_check_func=_post_check_func)
    self.assertTrue(self._success)

  def test_dbmtl_variational_dropout(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_variational_dropout.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_dbmtl_variational_dropout_feature_num(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_variational_dropout_feature_num.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_essm_variational_dropout(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/esmm_variational_dropout_on_taobao.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_fm_variational_dropout(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fm_variational_dropout_on_taobao.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_deepfm_with_combo_feature_variational_dropout(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_combo_variational_dropout_on_avazu_ctr.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_dbmtl_sequence_variational_dropout(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_variational_dropout_on_sequence_feature_taobao.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_din_variational_dropout(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/din_varitional_dropout_on_taobao.config',
        self._test_dir,
        post_check_func=test_utils.test_feature_selection)
    self.assertTrue(self._success)

  def test_rocket_launching(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/rocket_launching.config', self._test_dir)
    self.assertTrue(self._success)

  def test_rocket_launching_feature_based(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/rocket_launching_feature_based.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_rocket_launching_with_rtp_input(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/rocket_launching_with_rtp_input.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_mmoe(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_mmoe_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_train_with_ps_worker(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_fit_on_eval(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao.config',
        self._test_dir,
        num_evaluator=1,
        fit_on_eval=True)
    self.assertTrue(self._success)

  def test_unbalance_data(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao_unblanace.config',
        self._test_dir,
        total_steps=0,
        num_epoch=1,
        num_evaluator=1)
    self.assertTrue(self._success)

  def test_train_with_ps_worker_with_evaluator(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao.config',
        self._test_dir,
        num_evaluator=1)
    self.assertTrue(self._success)
    final_export_dir = os.path.join(self._test_dir, 'train/export/final')
    all_saved_files = glob.glob(final_export_dir + '/*/saved_model.pb')
    logging.info('final_export_dir=%s all_saved_files=%s' %
                 (final_export_dir, ','.join(all_saved_files)))
    self.assertTrue(len(all_saved_files) == 1)

  def test_train_with_ps_worker_chief_redundant(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao_chief_redundant.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_deepfm_embed_input(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/deepfm_with_embed.config', self._test_dir)
    self.assertTrue(self._success)

  def test_multi_tower_embed_input(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_with_embed.config', self._test_dir)
    self.assertTrue(self._success)

  def test_tfrecord_input(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/deepfm_on_criteo_tfrecord.config', self._test_dir)
    self.assertTrue(self._success)

  def test_batch_tfrecord_input(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/deepfm_on_criteo_batch_tfrecord.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_autodis_embedding(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_on_criteo_with_autodis.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_periodic_embedding(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_on_criteo_with_periodic.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sample_weight(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_with_sample_weight.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dssm_sample_weight(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_with_sample_weight.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_neg_sampler_with_sample_weight(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_with_sample_weight.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(
      LooseVersion(tf.__version__) != LooseVersion('2.3.0'),
      'MultiWorkerMirroredStrategy need tf version == 2.3')
  def test_train_with_multi_worker_mirror(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_multi_worker_mirrored_strategy_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(
      LooseVersion(tf.__version__) != LooseVersion('2.3.0'),
      'MultiWorkerMirroredStrategy need tf version == 2.3')
  def test_train_mmoe_with_multi_worker_mirror(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/mmoe_mirrored_strategy_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_fg_dtype(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg_test_dtype.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(six.PY2, 'Only run in python3')
  def test_share_not_used(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/share_not_used.config', self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_autoint(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/autoint_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_dcn(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dcn_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_dssm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_esmm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/esmm_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_mmoe(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/mmoe_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_ple(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/ple_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_rocket_launching(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/rocket_launching_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_simple_multi_task(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/simple_multi_task_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_sequence_wide_and_deep(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/wide_and_deep_on_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_numeric_boundary_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_numeric_boundary_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_numeric_hash_bucket_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_numeric_hash_bucket_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_numeric_raw_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_numeric_raw_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_numeric_num_buckets_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_numeric_num_buckets_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_numeric_boundary_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_numeric_boundary_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_numeric_hash_bucket_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_numeric_hash_bucket_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_numeric_raw_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_numeric_raw_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_numeric_num_buckets_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_numeric_num_buckets_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_sequence_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_sequence_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_multi_optimizer(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/wide_and_deep_two_opti.config', self._test_dir)
    self.assertTrue(self._success)

  def test_embedding_separate_optimizer(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_embed_adagrad.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_expr_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao_for_expr.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_gzip_data(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/din_on_gzip_data.config', self._test_dir)
    self.assertTrue(self._success)

  def test_cmd_config_param(self):

    def _post_check_config(pipeline_config):
      train_saved_config_path = os.path.join(self._test_dir,
                                             'train/pipeline.config')
      pipeline_config = config_util.get_configs_from_pipeline_file(
          train_saved_config_path)
      assert pipeline_config.model_config.deepfm.wide_output_dim == 8,\
          'invalid model_config.deepfm.wide_output_dim=%d' % \
          pipeline_config.model_config.deepfm.wide_output_dim

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir,
        post_check_func=_post_check_config,
        extra_cmd_args='--model_config.deepfm.wide_output_dim 8')

  def test_cmd_config_param_v2(self):

    def _post_check_config(pipeline_config):
      train_saved_config_path = os.path.join(self._test_dir,
                                             'train/pipeline.config')
      pipeline_config = config_util.get_configs_from_pipeline_file(
          train_saved_config_path)
      assert pipeline_config.model_config.deepfm.wide_output_dim == 1,\
          'invalid model_config.deepfm.wide_output_dim=%d' % \
          pipeline_config.model_config.deepfm.wide_output_dim

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir,
        post_check_func=_post_check_config,
        extra_cmd_args='--model_config.deepfm.wide_output_dim=1')

  def test_cmd_config_param_v3(self):

    def _post_check_config(pipeline_config):
      train_saved_config_path = os.path.join(self._test_dir,
                                             'train/pipeline.config')
      pipeline_config = config_util.get_configs_from_pipeline_file(
          train_saved_config_path)
      assert pipeline_config.model_config.deepfm.wide_output_dim == 3,\
          'invalid model_config.deepfm.wide_output_dim=%d' % \
          pipeline_config.model_config.deepfm.wide_output_dim

    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        self._test_dir,
        post_check_func=_post_check_config,
        extra_cmd_args='--model_config.deepfm.wide_output_dim="3"')

  def test_distribute_eval_deepfm_multi_cls(self):
    cur_eval_path = 'data/test/distribute_eval_test/deepfm_distribute_eval_dwd_avazu_out_multi_cls'
    self._success = test_utils.test_distributed_eval(
        'samples/model_config/deepfm_distribute_eval_multi_cls_on_avazu_ctr.config',
        cur_eval_path, self._test_dir)
    self.assertTrue(self._success)

  def test_distribute_eval_deepfm_single_cls(self):
    cur_eval_path = 'data/test/distribute_eval_test/dwd_distribute_eval_avazu_out_test_combo'
    self._success = test_utils.test_distributed_eval(
        'samples/model_config/deepfm_distribute_eval_combo_on_avazu_ctr.config',
        cur_eval_path, self._test_dir)
    self.assertTrue(self._success)

  def test_distribute_eval_dssm_pointwise_classification(self):
    cur_eval_path = 'data/test/distribute_eval_test/dssm_distribute_eval_pointwise_classification_taobao_ckpt'
    self._success = test_utils.test_distributed_eval(
        'samples/model_config/dssm_distribute_eval_pointwise_classification_on_taobao.config',
        cur_eval_path, self._test_dir)
    self.assertTrue(self._success)

  def test_distribute_eval_dssm_reg(self):
    cur_eval_path = 'data/test/distribute_eval_test/dssm_distribute_eval_reg_taobao_ckpt'
    self._success = test_utils.test_distributed_eval(
        'samples/model_config/dssm_distribute_eval_reg_on_taobao.config',
        cur_eval_path, self._test_dir)
    self.assertTrue(self._success)

  def test_distribute_eval_dropout(self):
    cur_eval_path = 'data/test/distribute_eval_test/dropoutnet_distribute_eval_taobao_ckpt'
    self._success = test_utils.test_distributed_eval(
        'samples/model_config/dropoutnet_distribute_eval_on_taobao.config',
        cur_eval_path, self._test_dir)
    self.assertTrue(self._success)

  def test_distribute_eval_esmm(self):
    cur_eval_path = 'data/test/distribute_eval_test/esmm_distribute_eval_taobao_ckpt'
    self._success = test_utils.test_distributed_eval(
        'samples/model_config/esmm_distribute_eval_on_taobao.config',
        cur_eval_path, self._test_dir)
    self.assertTrue(self._success)

  def test_share_no_used(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/share_embedding_not_used.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_neg_sampler_sequence_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_sequence_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_dssm_neg_sampler_need_key_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_need_key_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_on_multi_numeric_boundary_need_key_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_numeric_boundary_need_key_feature_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_on_multi_numeric_boundary_allow_key_transform(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_multi_numeric_boundary_allow_key_transform.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl_on_multi_numeric_boundary_aux_hist_seq(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_numeric_boundary_sequence_feature_aux_hist_seq_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_multi_tower_recall_neg_sampler_sequence_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_recall_neg_sampler_sequence_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(gl is None, 'graphlearn is not installed')
  def test_multi_tower_recall_neg_sampler_only_sequence_feature(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_recall_neg_sampler_only_sequence_feature.config',
        self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(hvd is None, 'horovod is not installed')
  def test_horovod(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/deepfm_combo_on_avazu_ctr.config',
        self._test_dir,
        use_hvd=True)
    self.assertTrue(self._success)

  @unittest.skipIf(hvd is None or sok is None,
                   'horovod and sok is not installed')
  def test_sok(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao_sok.config',
        self._test_dir,
        use_hvd=True)
    self.assertTrue(self._success)

  @unittest.skipIf(
      six.PY2 or tf_version.split('.')[0] != '2',
      'only run on python3 and tf 2.x')
  def test_train_parquet(self):
    os.environ[constant.NO_ARITHMETRIC_OPTI] = '1'
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dlrm_on_criteo_parquet.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(hvd is None, 'horovod is not installed')
  def test_train_parquet_embedding_parallel(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/dlrm_on_criteo_parquet_ep.config',
        self._test_dir,
        use_hvd=True)
    self.assertTrue(self._success)

  @unittest.skipIf(hvd is None, 'horovod is not installed')
  def test_train_parquet_embedding_parallel_v2(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/dlrm_on_criteo_parquet_ep_v2.config',
        self._test_dir,
        use_hvd=True)
    self.assertTrue(self._success)

  def test_pdn(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/pdn_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)


if __name__ == '__main__':
  tf.test.main()
