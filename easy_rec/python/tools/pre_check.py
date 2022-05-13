# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils import config_util
from easy_rec.python.utils import fg_util
from easy_rec.python.utils.check_utils import check_env_and_input_path
from easy_rec.python.utils.check_utils import check_sequence

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)
tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_multi_string(
    'data_input_path', None, help='data input path')

FLAGS = tf.app.flags.FLAGS


def _get_input_fn(data_config,
                  feature_configs,
                  data_path=None,
                  export_config=None):
  """Build estimator input function.

  Args:
    data_config:  dataset config
    feature_configs: FeatureConfig
    data_path: input_data_path
    export_config: configuration for exporting models,
      only used to build input_fn when exporting models

  Returns:
    subclass of Input
  """
  input_class_map = {y: x for x, y in data_config.InputType.items()}
  input_cls_name = input_class_map[data_config.input_type]

  input_class = Input.create_class(input_cls_name)
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    worker_num = len(tf_config['cluster']['worker'])
    task_index = tf_config['task']['index']
  else:
    worker_num = 1
    task_index = 0

  input_obj = input_class(
      data_config,
      feature_configs,
      data_path,
      task_index=task_index,
      task_num=worker_num,
      check_mode=True)
  input_fn = input_obj.create_input(export_config)
  return input_fn


def loda_pipeline_config(pipeline_config_path):
  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path, False)
  if pipeline_config.fg_json_path:
    fg_util.load_fg_json_to_config(pipeline_config)
  config_util.auto_expand_share_feature_configs(pipeline_config)
  return pipeline_config


def run_check(pipeline_config, input_path):
  logging.info('data_input_path: %s' % input_path)
  check_env_and_input_path(pipeline_config, input_path)
  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
  eval_input_fn = _get_input_fn(pipeline_config.data_config, feature_configs,
                                input_path)
  eval_spec = tf.estimator.EvalSpec(
      name='val',
      input_fn=eval_input_fn,
      steps=None,
      throttle_secs=10,
      exporters=[])
  input_iter = eval_spec.input_fn(
      mode=tf.estimator.ModeKeys.EVAL).make_one_shot_iterator()
  with tf.Session() as sess:
    try:
      while (True):
        input_feas, input_lbls = input_iter.get_next()
        features = sess.run(input_feas)
        check_sequence(pipeline_config, features)
    except tf.errors.OutOfRangeError:
      logging.info('pre-check finish...')


def main(argv):
  assert FLAGS.pipeline_config_path, 'pipeline_config_path should not be empty when checking!'
  pipeline_config = loda_pipeline_config(FLAGS.pipeline_config_path)

  if FLAGS.data_input_path:
    input_path = ','.join(FLAGS.data_input_path)
  else:
    assert pipeline_config.train_input_path or pipeline_config.eval_input_path, \
        'input_path should not be empty when checking!'
    input_path = pipeline_config.train_input_path + ',' + pipeline_config.eval_input_path

  run_check(pipeline_config, input_path)


if __name__ == '__main__':
  tf.app.run()
