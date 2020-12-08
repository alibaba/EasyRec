# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.main import export

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)
tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'checkpoint to be exported')
tf.app.flags.DEFINE_string('export_dir', None,
                           'directory where model should be exported to')

tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('export_dir')
FLAGS = tf.app.flags.FLAGS


def main(argv):
  export(FLAGS.export_dir, FLAGS.pipeline_config_path, FLAGS.checkpoint_path)


if __name__ == '__main__':
  tf.app.run()
