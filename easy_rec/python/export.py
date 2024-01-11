# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile

from easy_rec.python.main import export
from easy_rec.python.protos.train_pb2 import DistributionStrategy
from easy_rec.python.utils import config_util
from easy_rec.python.utils import estimator_utils

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

tf.app.flags.DEFINE_string('redis_url', None, 'export to redis url, host:port')
tf.app.flags.DEFINE_string('redis_passwd', None, 'export to redis passwd')
tf.app.flags.DEFINE_integer('redis_threads', 0, 'export to redis threads')
tf.app.flags.DEFINE_integer('redis_batch_size', 256,
                            'export to redis batch_size')
tf.app.flags.DEFINE_integer('redis_timeout', 600,
                            'export to redis time_out in seconds')
tf.app.flags.DEFINE_integer('redis_expire', 24,
                            'export to redis expire time in hour')
tf.app.flags.DEFINE_string('redis_embedding_version', '',
                           'redis embedding version')
tf.app.flags.DEFINE_integer('redis_write_kv', 1,
                            'whether to write embedding to redis')

tf.app.flags.DEFINE_string(
    'oss_path', None, 'write embed objects to oss folder, oss://bucket/folder')
tf.app.flags.DEFINE_string('oss_endpoint', None, 'oss endpoint')
tf.app.flags.DEFINE_string('oss_ak', None, 'oss ak')
tf.app.flags.DEFINE_string('oss_sk', None, 'oss sk')
tf.app.flags.DEFINE_integer('oss_threads', 10,
                            '# threads access oss at the same time')
tf.app.flags.DEFINE_integer('oss_timeout', 10,
                            'connect to oss, time_out in seconds')
tf.app.flags.DEFINE_integer('oss_expire', 24, 'oss expire time in hours')
tf.app.flags.DEFINE_integer('oss_write_kv', 1,
                            'whether to write embedding to oss')
tf.app.flags.DEFINE_string('oss_embedding_version', '', 'oss embedding version')

tf.app.flags.DEFINE_string('asset_files', '', 'more files to add to asset')
tf.app.flags.DEFINE_bool('verbose', False, 'print more debug information')

tf.app.flags.DEFINE_string('model_dir', None, help='will update the model_dir')
tf.app.flags.mark_flag_as_required('export_dir')

tf.app.flags.DEFINE_bool('clear_export', False, 'remove export_dir if exists')
tf.app.flags.DEFINE_string('export_done_file', '',
                           'a flag file to signal that export model is done')
FLAGS = tf.app.flags.FLAGS


def main(argv):
  extra_params = {}
  if FLAGS.redis_url:
    extra_params['redis_url'] = FLAGS.redis_url
  if FLAGS.redis_passwd:
    extra_params['redis_passwd'] = FLAGS.redis_passwd
  if FLAGS.redis_threads > 0:
    extra_params['redis_threads'] = FLAGS.redis_threads
  if FLAGS.redis_batch_size > 0:
    extra_params['redis_batch_size'] = FLAGS.redis_batch_size
  if FLAGS.redis_expire > 0:
    extra_params['redis_expire'] = FLAGS.redis_expire
  if FLAGS.redis_embedding_version:
    extra_params['redis_embedding_version'] = FLAGS.redis_embedding_version
  if FLAGS.redis_write_kv == 0:
    extra_params['redis_write_kv'] = False
  else:
    extra_params['redis_write_kv'] = True

  assert FLAGS.model_dir or FLAGS.pipeline_config_path, 'At least one of model_dir and pipeline_config_path exists.'
  if FLAGS.model_dir:
    pipeline_config_path = os.path.join(FLAGS.model_dir, 'pipeline.config')
    if file_io.file_exists(pipeline_config_path):
      logging.info('update pipeline_config_path to %s' % pipeline_config_path)
    else:
      pipeline_config_path = FLAGS.pipeline_config_path
  else:
    pipeline_config_path = FLAGS.pipeline_config_path

  if FLAGS.oss_path:
    extra_params['oss_path'] = FLAGS.oss_path
  if FLAGS.oss_endpoint:
    extra_params['oss_endpoint'] = FLAGS.oss_endpoint
  if FLAGS.oss_ak:
    extra_params['oss_ak'] = FLAGS.oss_ak
  if FLAGS.oss_sk:
    extra_params['oss_sk'] = FLAGS.oss_sk
  if FLAGS.oss_timeout > 0:
    extra_params['oss_timeout'] = FLAGS.oss_timeout
  if FLAGS.oss_expire > 0:
    extra_params['oss_expire'] = FLAGS.oss_expire
  if FLAGS.oss_threads > 0:
    extra_params['oss_threads'] = FLAGS.oss_threads
  if FLAGS.oss_write_kv:
    extra_params['oss_write_kv'] = True if FLAGS.oss_write_kv == 1 else False
  if FLAGS.oss_embedding_version:
    extra_params['oss_embedding_version'] = FLAGS.oss_embedding_version

  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  if pipeline_config.train_config.train_distribute in [
      DistributionStrategy.HorovodStrategy,
  ]:
    estimator_utils.init_hvd()
  elif pipeline_config.train_config.train_distribute in [
      DistributionStrategy.EmbeddingParallelStrategy,
      DistributionStrategy.SokStrategy
  ]:
    estimator_utils.init_hvd()
    estimator_utils.init_sok()

  if FLAGS.clear_export:
    logging.info('will clear export_dir=%s' % FLAGS.export_dir)
    if gfile.IsDirectory(FLAGS.export_dir):
      gfile.DeleteRecursively(FLAGS.export_dir)

  export_out_dir = export(FLAGS.export_dir, pipeline_config_path,
                          FLAGS.checkpoint_path, FLAGS.asset_files,
                          FLAGS.verbose, **extra_params)

  if FLAGS.export_done_file:
    flag_file = os.path.join(export_out_dir, FLAGS.export_done_file)
    logging.info('create export done file: %s' % flag_file)
    with gfile.GFile(flag_file, 'w') as fout:
      fout.write('ExportDone')


if __name__ == '__main__':
  tf.app.run()
