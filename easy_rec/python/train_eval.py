# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

import tensorflow as tf

from easy_rec.python.main import _train_and_evaluate_impl
from easy_rec.python.utils import config_util
from easy_rec.python.utils import hpo_util

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
tf.app.flags.DEFINE_string(
    'edit_config_json',
    None,
    help='edit pipeline config str, example: {"model_dir":"experiments/",'
    '"feature_config[0].boundaries":[4,5,6,7]}')

FLAGS = tf.app.flags.FLAGS


def main(argv):
  if FLAGS.pipeline_config_path is not None:
    pipeline_config = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path, False)
    if FLAGS.model_dir:
      pipeline_config.model_dir = FLAGS.model_dir
      logging.info('update model_dir to %s' % pipeline_config.model_dir)
    if FLAGS.hpo_param_path:
      with tf.gfile.GFile(FLAGS.hpo_param_path, 'r') as fin:
        hpo_config = json.load(fin)
        hpo_params = hpo_config['param']
        config_util.edit_config(pipeline_config, hpo_params)
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train)
      hpo_util.save_eval_metrics(
          pipeline_config.model_dir,
          metric_save_path=FLAGS.hpo_metric_save_path,
          has_evaluator=False)
    elif FLAGS.edit_config_json:
      config_json = json.loads(FLAGS.edit_config_json)
      config_util.edit_config(pipeline_config, config_json)
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train)
    else:
      config_util.auto_expand_share_feature_configs(pipeline_config)
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train)
  else:
    raise ValueError('pipeline_config_path should not be empty when training!')


if __name__ == '__main__':
  tf.app.run()
