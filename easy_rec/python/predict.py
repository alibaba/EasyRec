# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging

import tensorflow as tf

from easy_rec.python.main import predict

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
    'input_path', None, 'predict data path, if specified will '
    'override pipeline_config.eval_input_path')
tf.app.flags.DEFINE_string('output_path', None, 'path to save predict result')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
FLAGS = tf.app.flags.FLAGS


def main(argv):
  pred_result = predict(FLAGS.pipeline_config_path, FLAGS.checkpoint_path,
                        FLAGS.input_path)
  if FLAGS.output_path is not None:
    logging.info('will save predict result to %s' % FLAGS.output_path)
    with tf.gfile.GFile(FLAGS.output_path, 'wb') as fout:
      for k in pred_result:
        fout.write(str(k).replace("'", '"') + '\n')


if __name__ == '__main__':
  tf.app.run()
