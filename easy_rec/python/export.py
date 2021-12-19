# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from easy_rec.python.main import export

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

logging.basicConfig(format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s', level=logging.INFO)
tf.app.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config ' 'file.')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'checkpoint to be exported')
tf.app.flags.DEFINE_string('export_dir', None, 'directory where model should be exported to')
tf.app.flags.DEFINE_string('redis_url', None, 'export to redis url, host:port')
tf.app.flags.DEFINE_string('redis_passwd', None, 'export to redis passwd')
tf.app.flags.DEFINE_integer('redis_threads', 0, 'export to redis threads')
tf.app.flags.DEFINE_integer('redis_batch_size', 256, 'export to redis batch_size')
tf.app.flags.DEFINE_integer('redis_timeout', 600, 'export to redis time_out in seconds')
tf.app.flags.DEFINE_integer('redis_expire', 24, 'export to redis expire time in hour')
tf.app.flags.DEFINE_string('redis_embedding_version', '', 'redis embedding version')
tf.app.flags.DEFINE_integer('redis_write_kv', 1, 'whether to write embedding to redis')
tf.app.flags.DEFINE_string('asset_files', '', 'more files to add to asset')
tf.app.flags.DEFINE_bool('verbose', False, 'print more debug information')

tf.app.flags.DEFINE_string('model_dir', None, help='will update the model_dir')
tf.app.flags.mark_flag_as_required('export_dir')
FLAGS = tf.app.flags.FLAGS


def main(argv):
    redis_params = {}
    if FLAGS.redis_url:
        redis_params['redis_url'] = FLAGS.redis_url
    if FLAGS.redis_passwd:
        redis_params['redis_passwd'] = FLAGS.redis_passwd
    if FLAGS.redis_threads > 0:
        redis_params['redis_threads'] = FLAGS.redis_threads
    if FLAGS.redis_batch_size > 0:
        redis_params['redis_batch_size'] = FLAGS.redis_batch_size
    if FLAGS.redis_expire > 0:
        redis_params['redis_expire'] = FLAGS.redis_expire
    if FLAGS.redis_embedding_version:
        redis_params['redis_embedding_version'] = FLAGS.redis_embedding_version
    if FLAGS.redis_write_kv == 0:
        redis_params['redis_write_kv'] = False
    else:
        redis_params['redis_write_kv'] = True

    assert FLAGS.model_dir or FLAGS.pipeline_config_path, 'At least one of model_dir and pipeline_config_path exists.'
    if FLAGS.model_dir:
        pipeline_config_path = os.path.join(FLAGS.model_dir, 'pipeline.config')
        if file_io.file_exists(pipeline_config_path):
            logging.info('update pipeline_config_path to %s' % pipeline_config_path)
        else:
            pipeline_config_path = FLAGS.pipeline_config_path
    else:
        pipeline_config_path = FLAGS.pipeline_config_path

    export(
        FLAGS.export_dir, pipeline_config_path, FLAGS.checkpoint_path, FLAGS.asset_files, FLAGS.verbose, **redis_params
    )


if __name__ == '__main__':
    tf.app.run()
