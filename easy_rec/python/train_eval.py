# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import logging
import os

import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.main import _train_and_evaluate_impl
from easy_rec.python.utils import config_util
from easy_rec.python.utils import ds_util
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import fg_util
from easy_rec.python.utils import hpo_util
from easy_rec.python.utils.config_util import process_neg_sampler_data_path
from easy_rec.python.utils.config_util import set_eval_input_path
from easy_rec.python.utils.config_util import set_train_input_path

from easy_rec.python.utils.distribution_utils import set_tf_config_and_get_train_worker_num_on_ds  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--pipeline_config_path',
      type=str,
      default=None,
      help='Path to pipeline config file.')
  parser.add_argument(
      '--continue_train',
      action='store_true',
      default=False,
      help='continue train using existing model_dir')
  parser.add_argument(
      '--hpo_param_path',
      type=str,
      default=None,
      help='hyperparam tuning param path')
  parser.add_argument(
      '--hpo_metric_save_path',
      type=str,
      default=None,
      help='hyperparameter save metric path')
  parser.add_argument(
      '--model_dir',
      type=str,
      default=None,
      help='will update the model_dir in pipeline_config')
  parser.add_argument(
      '--train_input_path',
      type=str,
      nargs='*',
      default=None,
      help='train data input path')
  parser.add_argument(
      '--eval_input_path',
      type=str,
      nargs='*',
      default=None,
      help='eval data input path')
  parser.add_argument(
      '--fine_tune_checkpoint',
      type=str,
      default=None,
      help='will update the train_config.fine_tune_checkpoint in pipeline_config'
  )
  parser.add_argument(
      '--edit_config_json',
      type=str,
      default=None,
      help='edit pipeline config str, example: {"model_dir":"experiments/",'
      '"feature_config.feature[0].boundaries":[4,5,6,7]}')
  parser.add_argument(
      '--ignore_finetune_ckpt_error',
      action='store_true',
      default=False,
      help='During incremental training, ignore the problem of missing fine_tune_checkpoint files'
  )
  parser.add_argument(
      '--odps_config', type=str, default=None, help='odps config path')
  parser.add_argument(
      '--is_on_ds', action='store_true', default=False, help='is on ds')
  parser.add_argument(
      '--check_mode',
      action='store_true',
      default=False,
      help='is use check mode')
  parser.add_argument(
      '--selected_cols', type=str, default=None, help='select input columns')
  args, extra_args = parser.parse_known_args()

  edit_config_json = {}
  if args.edit_config_json:
    edit_config_json = json.loads(args.edit_config_json)

  if extra_args is not None and len(extra_args) > 0:
    config_util.parse_extra_config_param(extra_args, edit_config_json)

  if args.pipeline_config_path is not None:
    pipeline_config = config_util.get_configs_from_pipeline_file(
        args.pipeline_config_path, False)
    if args.selected_cols:
      pipeline_config.data_config.selected_cols = args.selected_cols
    if args.model_dir:
      pipeline_config.model_dir = args.model_dir
      logging.info('update model_dir to %s' % pipeline_config.model_dir)
    if args.train_input_path:
      set_train_input_path(pipeline_config, args.train_input_path)
    if args.eval_input_path:
      set_eval_input_path(pipeline_config, args.eval_input_path)

    if args.fine_tune_checkpoint:
      ckpt_path = estimator_utils.get_latest_checkpoint_from_checkpoint_path(
          args.fine_tune_checkpoint, args.ignore_finetune_ckpt_error)

      if ckpt_path:
        pipeline_config.train_config.fine_tune_checkpoint = ckpt_path

    if pipeline_config.fg_json_path:
      fg_util.load_fg_json_to_config(pipeline_config)

    if args.odps_config:
      os.environ['ODPS_CONFIG_FILE_PATH'] = args.odps_config

    if len(edit_config_json) > 0:
      fine_tune_checkpoint = edit_config_json.get('train_config', {}).get(
          'fine_tune_checkpoint', None)
      if fine_tune_checkpoint:
        ckpt_path = estimator_utils.get_latest_checkpoint_from_checkpoint_path(
            args.fine_tune_checkpoint, args.ignore_finetune_ckpt_error)
        edit_config_json['train_config']['fine_tune_checkpoint'] = ckpt_path
      config_util.edit_config(pipeline_config, edit_config_json)

    process_neg_sampler_data_path(pipeline_config)

    if args.is_on_ds:
      ds_util.set_on_ds()
      set_tf_config_and_get_train_worker_num_on_ds()
      if pipeline_config.train_config.fine_tune_checkpoint:
        ds_util.cache_ckpt(pipeline_config)

    if args.hpo_param_path:
      with gfile.GFile(args.hpo_param_path, 'r') as fin:
        hpo_config = json.load(fin)
        hpo_params = hpo_config['param']
        config_util.edit_config(pipeline_config, hpo_params)
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, args.continue_train,
                               args.check_mode)
      hpo_util.save_eval_metrics(
          pipeline_config.model_dir,
          metric_save_path=args.hpo_metric_save_path,
          has_evaluator=False)
    else:
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, args.continue_train,
                               args.check_mode)
  else:
    raise ValueError('pipeline_config_path should not be empty when training!')
