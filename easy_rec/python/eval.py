# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging

import six
import tensorflow as tf

from easy_rec.python.main import evaluate

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None, 'checkpoint to be evaled '
    ' if not specified, use the latest checkpoint in '
    'train_config.model_dir')
tf.app.flags.DEFINE_string(
    'eval_input_path', None, 'eval data path, if specified will '
    'override pipeline_config.eval_input_path')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
FLAGS = tf.app.flags.FLAGS


def main(argv):
  eval_result = evaluate(FLAGS.pipeline_config_path, FLAGS.checkpoint_path,
                         FLAGS.eval_input_path)
  for key in sorted(eval_result):
    # skip logging binary data
    if isinstance(eval_result[key], six.binary_type):
      continue
    logging.info('%s: %s' % (key, str(eval_result[key])))


if __name__ == '__main__':
  tf.app.run()
