# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import six
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from easy_rec.python.main import evaluate

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

logging.basicConfig(format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s', level=logging.INFO)

tf.app.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config ' 'file.')
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    None,
    'checkpoint to be evaled ' ' if not specified, use the latest checkpoint in ' 'train_config.model_dir',
)
tf.app.flags.DEFINE_multi_string(
    'eval_input_path', None, 'eval data path, if specified will ' 'override pipeline_config.eval_input_path'
)
tf.app.flags.DEFINE_string('model_dir', None, help='will update the model_dir')
tf.app.flags.DEFINE_string('odps_config', None, help='odps config path')
FLAGS = tf.app.flags.FLAGS


def main(argv):
    if FLAGS.odps_config:
        os.environ['ODPS_CONFIG_FILE_PATH'] = FLAGS.odps_config

    assert FLAGS.model_dir or FLAGS.pipeline_config_path, 'At least one of model_dir and pipeline_config_path exists.'
    if FLAGS.model_dir:
        pipeline_config_path = os.path.join(FLAGS.model_dir, 'pipeline.config')
        if file_io.file_exists(pipeline_config_path):
            logging.info('update pipeline_config_path to %s' % pipeline_config_path)
        else:
            pipeline_config_path = FLAGS.pipeline_config_path
    else:
        pipeline_config_path = FLAGS.pipeline_config_path

    eval_result = evaluate(pipeline_config_path, FLAGS.checkpoint_path, FLAGS.eval_input_path)
    for key in sorted(eval_result):
        # skip logging binary data
        if isinstance(eval_result[key], six.binary_type):
            continue
        logging.info('%s: %s' % (key, str(eval_result[key])))


if __name__ == '__main__':
    tf.app.run()
