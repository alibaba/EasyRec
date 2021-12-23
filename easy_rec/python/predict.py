# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import os

import tensorflow as tf

from easy_rec.python.inference.predictor import Predictor

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

tf.app.flags.DEFINE_string(
    'checkpoint_path', None, 'checkpoint to be evaled '
    ' if not specified, use the latest checkpoint in '
    'train_config.model_dir')
tf.app.flags.DEFINE_string(
    'input_path', None, 'predict data path, if specified will '
    'override pipeline_config.eval_input_path')
tf.app.flags.DEFINE_string('output_path', None, 'path to save predict result')

tf.app.flags.DEFINE_string('saved_model_dir', None, help='save model dir')
tf.app.flags.DEFINE_integer('batch_size', 1024, help='batch size')
tf.app.flags.DEFINE_string(
    'reserved_cols', 'ALL_COLUMNS',
    'columns to keep from input table,  they are separated with ,')
tf.app.flags.DEFINE_string(
    'output_cols', 'ALL_COLUMNS',
    'output columns, such as: score float. multiple columns are separated by ,')
tf.app.flags.DEFINE_string('input_sep', ',', 'separator of predict result file')
tf.app.flags.DEFINE_string('output_sep', chr(1),
                           'separator of predict result file')
FLAGS = tf.app.flags.FLAGS


def main(argv):

  predictor = Predictor(FLAGS.saved_model_dir)
  logging.info('input_path = %s, output_path = %s' %
               (FLAGS.input_path, FLAGS.output_path))
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    worker_num = len(tf_config['cluster']['worker'])
    task_index = tf_config['task']['index']
  else:
    worker_num = 1
    task_index = 0
  predictor.predict_impl(
      FLAGS.input_path,
      FLAGS.output_path,
      reserved_cols=FLAGS.reserved_cols,
      output_cols=FLAGS.output_cols,
      slice_id=task_index,
      slice_num=worker_num,
      input_sep=FLAGS.input_sep,
      output_sep=FLAGS.output_sep)


if __name__ == '__main__':
  tf.app.run()
