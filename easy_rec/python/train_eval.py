# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

import tensorflow as tf

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
tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_bool('continue_train', False,
                         'continue train using existing '
                         'model dir')
tf.app.flags.DEFINE_string(
    'hpo_param_path', None, help='hyperparam tuning param path')
tf.app.flags.DEFINE_string(
    'hpo_metric_save_path', None, help='hyperparameter save metric path')
tf.app.flags.DEFINE_string(
    'model_dir', None, help='will update the model_dir in pipeline_config')
tf.app.flags.DEFINE_multi_string(
    'train_input_path', None, help='train data input path')
tf.app.flags.DEFINE_multi_string(
    'eval_input_path', None, help='eval data input path')
tf.app.flags.DEFINE_string(
    'fine_tune_checkpoint',
    None,
    help='will update the train_config.fine_tune_checkpoint in pipeline_config')
tf.app.flags.DEFINE_string(
    'edit_config_json',
    None,
    help='edit pipeline config str, example: {"model_dir":"experiments/",'
    '"feature_config.feature[0].boundaries":[4,5,6,7]}')
tf.app.flags.DEFINE_bool(
    'ignore_finetune_ckpt_error', False,
    'During incremental training, ignore the problem of missing fine_tune_checkpoint files'
)
tf.app.flags.DEFINE_string('odps_config', None, help='odps config path')
tf.app.flags.DEFINE_bool('is_on_ds', False, help='is on ds')
tf.app.flags.DEFINE_bool('check_mode', False, help='is use check mode')
tf.app.flags.DEFINE_string('selected_cols', None, help='')
FLAGS = tf.app.flags.FLAGS


def main(argv):
  if FLAGS.pipeline_config_path is not None:
    pipeline_config = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path, False)
    if FLAGS.selected_cols:
      pipeline_config.data_config.selected_cols = FLAGS.selected_cols
    if FLAGS.model_dir:
      pipeline_config.model_dir = FLAGS.model_dir
      logging.info('update model_dir to %s' % pipeline_config.model_dir)
    if FLAGS.train_input_path:
      set_train_input_path(pipeline_config, FLAGS.train_input_path)
    if FLAGS.eval_input_path:
      set_eval_input_path(pipeline_config, FLAGS.eval_input_path)

    if FLAGS.fine_tune_checkpoint:
      ckpt_path = estimator_utils.get_latest_checkpoint_from_checkpoint_path(
          FLAGS.fine_tune_checkpoint, FLAGS.ignore_finetune_ckpt_error)

      if ckpt_path:
        pipeline_config.train_config.fine_tune_checkpoint = ckpt_path

    if pipeline_config.fg_json_path:
      fg_util.load_fg_json_to_config(pipeline_config)

    if FLAGS.odps_config:
      os.environ['ODPS_CONFIG_FILE_PATH'] = FLAGS.odps_config

    if FLAGS.edit_config_json:
      config_json = json.loads(FLAGS.edit_config_json)
      fine_tune_checkpoint = config_json.get('train_config',
                                             {}).get('fine_tune_checkpoint',
                                                     None)
      if fine_tune_checkpoint:
        ckpt_path = estimator_utils.get_latest_checkpoint_from_checkpoint_path(
            FLAGS.fine_tune_checkpoint, FLAGS.ignore_finetune_ckpt_error)
        config_json['train_config']['fine_tune_checkpoint'] = ckpt_path
      config_util.edit_config(pipeline_config, config_json)

    process_neg_sampler_data_path(pipeline_config)

    if FLAGS.is_on_ds:
      ds_util.set_on_ds()
      set_tf_config_and_get_train_worker_num_on_ds()
      if pipeline_config.train_config.fine_tune_checkpoint:
        fine_tune_ckpt_path = pipeline_config.train_config.fine_tune_checkpoint
        if fine_tune_ckpt_path.startswith('hdfs://'):
          tmpdir = os.path.dirname(fine_tune_ckpt_path.replace('hdfs://', ''))
          tmpdir = os.path.join('/tmp/experiments', tmpdir)
          logging.info('will cache fine_tune_ckpt to local dir: %s' % tmpdir)
          if tf.gfile.IsDirectory(tmpdir):
            tf.gfile.DeleteRecursively(tmpdir)
          tf.gfile.MakeDirs(tmpdir)
          for src_path in tf.gfile.Glob(fine_tune_ckpt_path + '*'):
            dst_path = os.path.join(tmpdir, os.path.basename(src_path))
            logging.info('will copy %s to local path %s' % (src_path, dst_path))
            tf.gfile.Copy(src_path, dst_path, overwrite=True)
          ckpt_filename = os.path.basename(fine_tune_ckpt_path)
          fine_tune_ckpt_path = os.path.join(tmpdir, ckpt_filename)
        pipeline_config.train_config.fine_tune_checkpoint = fine_tune_ckpt_path
        logging.info('will restore from %s' % fine_tune_ckpt_path)

    if FLAGS.hpo_param_path:
      with tf.gfile.GFile(FLAGS.hpo_param_path, 'r') as fin:
        hpo_config = json.load(fin)
        hpo_params = hpo_config['param']
        config_util.edit_config(pipeline_config, hpo_params)
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train,
                               FLAGS.check_mode)
      hpo_util.save_eval_metrics(
          pipeline_config.model_dir,
          metric_save_path=FLAGS.hpo_metric_save_path,
          has_evaluator=False)
    else:
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train,
                               FLAGS.check_mode)
  else:
    raise ValueError('pipeline_config_path should not be empty when training!')


if __name__ == '__main__':
  tf.app.run()
