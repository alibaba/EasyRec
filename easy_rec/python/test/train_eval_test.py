# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import logging
import os
import unittest
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from easy_rec.python.main import predict
from easy_rec.python.utils import config_util
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import test_utils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
gfile = tf.gfile


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

  def test_dlrm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dlrm_on_taobao.config', self._test_dir)

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

  def test_multi_tower_gauc(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/multi_tower_on_taobao_gauc.config',
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
    self.assertAllClose(
        ckpts_times[1:] - ckpts_times[:-1], [20] * (len(ckpts_times) - 1),
        atol=8)
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
        total_steps=1000,
        post_check_func=_post_check_func)
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

  def test_din(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/din_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_bst(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/bst_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dcn(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dcn_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dcn_with_f1(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dcn_f1_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_autoint(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/autoint_on_taobao.config', self._test_dir)
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

  def test_dssm_neg_sampler(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dssm_neg_sampler_v2(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_neg_sampler_v2_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_dssm_hard_neg_sampler(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_hard_neg_sampler_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

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

  # def test_deepfm_with_sequence_attention(self):
  #   self._success = test_utils.test_single_train_eval(
  #       'samples/model_config/deppfm_seq_attn_on_taobao.config', self._test_dir)
  #   self.assertTrue(self._success)

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

  def test_tag_kv_input(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/kv_tag.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dbmtl(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_on_taobao.config', self._test_dir)
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

  def test_dbmtl_mmoe(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dbmtl_mmoe_on_taobao.config', self._test_dir)
    self.assertTrue(self._success)

  def test_train_with_ps_worker(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_on_taobao.config', self._test_dir)
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

  def test_sample_weight(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/deepfm_with_sample_weight.config', self._test_dir)
    self.assertTrue(self._success)

  def test_dssm_sample_weight(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/dssm_with_sample_weight.config', self._test_dir)
    self.assertTrue(self._success)

  @unittest.skipIf(
      LooseVersion(tf.__version__) < LooseVersion('2.3.0'),
      'MultiWorkerMirroredStrategy need tf version > 2.3')
  def test_train_with_multi_worker_mirror(self):
    self._success = test_utils.test_distributed_train_eval(
        'samples/model_config/multi_tower_multi_worker_mirrored_strategy_on_taobao.config',
        self._test_dir)
    self.assertTrue(self._success)

  def test_fg_dtype(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/taobao_fg_test_dtype.config', self._test_dir)
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

  def test_sequence_fm(self):
    self._success = test_utils.test_single_train_eval(
        'samples/model_config/fm_on_sequence_feature_taobao.config',
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


if __name__ == '__main__':
  tf.test.main()
