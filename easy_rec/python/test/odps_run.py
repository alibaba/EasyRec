# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import os
import shutil
import sys

import oss2
import tensorflow as tf

from easy_rec.python.test.odps_test_cls import OdpsTest
from easy_rec.python.test.odps_test_prepare import prepare
from easy_rec.python.test.odps_test_util import OdpsOSSConfig
from easy_rec.python.test.odps_test_util import delete_oss_path
from easy_rec.python.test.odps_test_util import get_oss_bucket
from easy_rec.python.utils import config_util

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

odps_oss_config = OdpsOSSConfig()


class TestPipelineOnOdps(tf.test.TestCase):
  """train eval export test on odps."""

  def test_deepfm(self):
    start_files = ['deep_fm/create_inner_deepfm_table.sql']
    test_files = [
        'deep_fm/train_deepfm_model.sql', 'deep_fm/eval_deepfm.sql',
        'deep_fm/export_deepfm.sql', 'deep_fm/predict_deepfm.sql',
        'deep_fm/export_rtp_ckpt.sql'
    ]
    end_file = ['deep_fm/drop_table.sql']

    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_mmoe(self):
    start_files = ['mmoe/create_inner_mmoe_table.sql']
    test_files = [
        'mmoe/train_mmoe_model.sql',
        'mmoe/eval_mmoe.sql',
        'mmoe/export_mmoe.sql',
        'mmoe/predict_mmoe.sql',
    ]
    end_file = ['mmoe/drop_mmoe_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_dssm(self):
    start_files = [
        'dssm/create_inner_dssm_table.sql',
    ]
    test_files = [
        'dssm/train_dssm_model.sql',
        'dssm/eval_dssm.sql',
        'dssm/export_dssm.sql',
        'dssm/predict_dssm.sql',
    ]
    end_file = [
        'dssm/drop_dssm_table.sql',
    ]
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_multi_tower(self):
    start_files = ['multi_tower/create_inner_multi_tower_table.sql']
    test_files = [
        'multi_tower/train_multil_tower_din_model.sql',
        'multi_tower/train_multil_tower_bst_model.sql',
        'multi_tower/eval_multil_tower.sql',
        'multi_tower/export_multil_tower.sql',
        'multi_tower/export_again_multi_tower.sql',
        'multi_tower/predict_multil_tower.sql',
    ]
    end_file = ['multi_tower/drop_multil_tower_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_other(self):
    start_files = ['deep_fm/create_inner_deepfm_table.sql']
    test_files = [
        # 'other_test/test_train_gpuRequired_mirrored', # 线上报错，
        # 'other_test/test_train_distribute_strategy_collective',  # 线上报错，
        'other_test/test_train_hpo_with_evaluator.sql',
        # 'other_test/test_train_version.sql',
        # 'other_test/test_train_distribute_strategy_ess.sql',
        'other_test/test_train_before_export.sql',
        'other_test/test_eval_checkpoint_path.sql',
        'other_test/test_export_checkpoint_path.sql',
        'other_test/test_export_update_model_dir.sql',
        'other_test/test_predict_selected_cols.sql',
    ]
    end_file = ['other_test/drop_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_best_exporter(self):
    start_files = ['deep_fm/create_inner_deepfm_table.sql']
    test_files = [
        'other_test/test_train_best_export.sql',
    ]
    end_file = ['other_test/drop_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    config_path = os.path.join(
        odps_oss_config.temp_dir,
        'configs/dwd_avazu_ctr_deepmodel_ext_best_export.config')
    config = config_util.get_configs_from_pipeline_file(config_path)
    model_dir = config.model_dir
    logging.info('raw model_dir = %s' % model_dir)
    if model_dir.startswith('oss://'):
      spos = model_dir.index('/', len('oss://') + 1) + 1
      model_dir = model_dir[spos:]
    logging.info('stripped model_dir = %s' % model_dir)

    bucket = get_oss_bucket(odps_oss_config.oss_key, odps_oss_config.oss_secret,
                            odps_oss_config.endpoint,
                            odps_oss_config.bucket_name)
    best_ckpt_prefix = os.path.join(model_dir, 'best_ckpt/model.ckpt')
    best_ckpts = [
        x.key
        for x in oss2.ObjectIterator(bucket, prefix=best_ckpt_prefix)
        if x.key.endswith('.meta')
    ]
    logging.info('best ckpts: %s' % str(best_ckpts))
    assert len(best_ckpts) <= 2, 'too many best ckpts: %s' % str(best_ckpts)
    best_export_prefix = os.path.join(model_dir, 'export/best/')
    best_exports = [
        x.key
        for x in oss2.ObjectIterator(bucket, prefix=best_export_prefix)
        if x.key.endswith('/saved_model.pb')
    ]
    logging.info('best exports: %s' % str(best_exports))
    assert len(
        best_exports) <= 2, 'too many best exports: %s' % str(best_exports)
    return True

  def test_embedding_variable(self):
    start_files = [
        'embedding_variable/create_table.sql',
    ]
    test_files = [
        'embedding_variable/train.sql', 'embedding_variable/train_work_que.sql',
        'embedding_variable/export.sql'
    ]
    end_file = ['embedding_variable/drop_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_multi_value_export(self):
    start_files = ['multi_value/create_inner_multi_value_table.sql']
    test_files = ['multi_value/train_multi_tower_model.sql']
    end_file = ['multi_value/drop_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_boundary_test(self):
    start_files = [
        'boundary/create_inner_boundary_table.sql',
    ]
    test_files = [
        'boundary/train_multi_tower_model.sql',
        'boundary/finetune_multi_tower_model.sql',
        'boundary/finetune_multi_tower_conti.sql', 'boundary/train_compat.sql'
    ]
    end_file = ['boundary/drop_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()

  def test_vector_retrieve(self):
    start_files = ['vector_retrieve/create_inner_vector_table.sql']
    test_files = ['vector_retrieve/run_vector_retrieve.sql']
    end_file = ['vector_retrieve/drop_table.sql']
    tot = OdpsTest(start_files, test_files, end_file, odps_oss_config)
    tot.start_test()
    tot.drop_table()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--odps_config', type=str, default=None, help='odps config path')
  parser.add_argument(
      '--oss_config', type=str, default=None, help='ossutilconfig path')
  parser.add_argument(
      '--bucket_name', type=str, default=None, help='test oss bucket name')
  parser.add_argument('--arn', type=str, default=None, help='oss rolearn')
  parser.add_argument(
      '--odpscmd', type=str, default='odpscmd', help='odpscmd path')
  parser.add_argument(
      '--algo_name',
      type=str,
      default='easy_rec_ext',
      help='whether use pai-tf 1.15')
  parser.add_argument(
      '--algo_project', type=str, default=None, help='algo project name')
  parser.add_argument(
      '--algo_res_project',
      type=str,
      default=None,
      help='algo resource project name')
  parser.add_argument(
      '--algo_version', type=str, default=None, help='algo version')
  parser.add_argument(
      '--is_outer',
      type=int,
      default=1,
      help='is outer pai or inner pai, the arguments are differed slightly due to history reasons'
  )
  args, unknown_args = parser.parse_known_args()
  sys.argv = [sys.argv[0]]
  for unk_arg in unknown_args:
    sys.argv.append(unk_arg)

  if args.odps_config:
    odps_oss_config.load_odps_config(args.odps_config)
  if args.oss_config:
    odps_oss_config.load_oss_config(args.oss_config)
  if args.odpscmd:
    odps_oss_config.odpscmd_path = args.odpscmd
  if args.algo_project:
    odps_oss_config.algo_project = args.algo_project
  if args.algo_res_project:
    odps_oss_config.algo_res_project = args.algo_res_project
  if args.algo_version:
    odps_oss_config.algo_version = args.algo_version
  algo_names = ['easy_rec_ext15', 'easy_rec_ext']
  assert args.algo_name in algo_names, 'algo_name must be oneof: %s' % (
      ','.join(algo_names))
  odps_oss_config.algo_name = args.algo_name
  if args.arn:
    odps_oss_config.arn = args.arn
  if args.bucket_name:
    odps_oss_config.bucket_name = args.bucket_name
  odps_oss_config.is_outer = args.is_outer

  prepare(odps_oss_config)
  tf.test.main()
  bucket = get_oss_bucket(odps_oss_config.oss_key, odps_oss_config.oss_secret,
                          odps_oss_config.endpoint, odps_oss_config.bucket_name)
  delete_oss_path(bucket, odps_oss_config.exp_dir, odps_oss_config.bucket_name)
  shutil.rmtree(odps_oss_config.temp_dir)
