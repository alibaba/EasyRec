# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import logging
# use few threads to avoid oss error
import os

import tensorflow as tf
import yaml
from tensorflow.python.platform import gfile

import easy_rec
from easy_rec.python.inference.predictor import ODPSPredictor
from easy_rec.python.inference.vector_retrieve import VectorRetrieve
from easy_rec.python.tools.pre_check import run_check
from easy_rec.python.utils import config_util
from easy_rec.python.utils import fg_util
from easy_rec.python.utils import hpo_util
from easy_rec.python.utils import pai_util
from easy_rec.python.utils.distribution_utils import DistributionStrategyMap
from easy_rec.python.utils.distribution_utils import set_distribution_config

from easy_rec.python.utils.distribution_utils import set_tf_config_and_get_train_worker_num  # NOQA
os.environ['OENV_MultiWriteThreadsNum'] = '4'
os.environ['OENV_MultiCopyThreadsNum'] = '4'

if not tf.__version__.startswith('1.12'):
  tf = tf.compat.v1
  try:
    import tensorflow_io as tfio  # noqa: F401
  except Exception as ex:
    logging.error('failed to import tfio: %s' % str(ex))
  tf.disable_eager_execution()

from easy_rec.python.main import _train_and_evaluate_impl as train_and_evaluate_impl  # NOQA

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

tf.app.flags.DEFINE_string('worker_hosts', '',
                           'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('job_name', '', 'task type, ps/worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_string('config', '', 'EasyRec config file path')
tf.app.flags.DEFINE_string('cmd', 'train',
                           'command type, train/evaluate/export')
tf.app.flags.DEFINE_string('tables', '', 'tables passed by pai command')

# flags for train
tf.app.flags.DEFINE_integer('num_gpus_per_worker', 1,
                            'number of gpu to use in training')
tf.app.flags.DEFINE_boolean('with_evaluator', False,
                            'whether a evaluator is necessary')
tf.app.flags.DEFINE_string(
    'eval_method', 'none', 'default to none, choices are [none: not evaluate,' +
    'master: evaluate on master, separate: evaluate on a separate task]')

tf.app.flags.DEFINE_string('distribute_strategy', '',
                           'training distribute strategy')
tf.app.flags.DEFINE_string('edit_config_json', '', 'edit config json string')
tf.app.flags.DEFINE_string('train_tables', '', 'tables used for train')
tf.app.flags.DEFINE_string('eval_tables', '', 'tables used for evaluation')
tf.app.flags.DEFINE_string('boundary_table', '', 'tables used for boundary')
tf.app.flags.DEFINE_string('sampler_table', '', 'tables used for sampler')
tf.app.flags.DEFINE_string('fine_tune_checkpoint', None,
                           'finetune checkpoint path')
tf.app.flags.DEFINE_string('query_table', '',
                           'table used for retrieve vector neighbours')
tf.app.flags.DEFINE_string('doc_table', '',
                           'table used for be retrieved as indexed vectors')
tf.app.flags.DEFINE_enum('knn_distance', 'inner_product',
                         ['l2', 'inner_product'], 'type of knn distance')
tf.app.flags.DEFINE_integer('knn_num_neighbours', None,
                            'top n neighbours to be retrieved')
tf.app.flags.DEFINE_integer('knn_feature_dims', None,
                            'number of feature dimensions')
tf.app.flags.DEFINE_enum(
    'knn_index_type', 'ivfflat',
    ['flat', 'ivfflat', 'ivfpq', 'gpu_flat', 'gpu_ivfflat', 'gpu_ivfpg'],
    'knn index type')
tf.app.flags.DEFINE_string('knn_feature_delimiter', ',',
                           'delimiter for feature vectors')
tf.app.flags.DEFINE_integer('knn_nlist', 5,
                            'number of split part on each worker')
tf.app.flags.DEFINE_integer('knn_nprobe', 2,
                            'number of probe part on each worker')
tf.app.flags.DEFINE_integer(
    'knn_compress_dim', 8,
    'number of dimensions after compress for `ivfpq` and `gpu_ivfpq`')

# flags used for evaluate & export
tf.app.flags.DEFINE_string(
    'checkpoint_path', '', 'checkpoint to be evaluated or exported '
    'if not specified, use the latest checkpoint '
    'in train_config.model_dir')
# flags used for evaluate
tf.app.flags.DEFINE_string('eval_result_path', 'eval_result.txt',
                           'eval result metric file')
tf.app.flags.DEFINE_bool('distribute_eval', False,
                         'use distribute parameter server for train and eval.')
# flags used for export
tf.app.flags.DEFINE_string('export_dir', '',
                           'directory where model should be exported to')
tf.app.flags.DEFINE_bool('clear_export', False, 'remove export_dir if exists')
tf.app.flags.DEFINE_boolean('continue_train', True,
                            'use the same model to continue train or not')

# flags used for predict
tf.app.flags.DEFINE_string('saved_model_dir', '',
                           'directory where saved_model.pb exists')
tf.app.flags.DEFINE_string('outputs', '', 'output tables')
tf.app.flags.DEFINE_string(
    'all_cols', '',
    'union of (selected_cols, reserved_cols), separated with , ')
tf.app.flags.DEFINE_string(
    'all_col_types', '',
    'column data types, for build record defaults, separated with ,')
tf.app.flags.DEFINE_string(
    'selected_cols', '',
    'columns to keep from input table,  they are separated with ,')
tf.app.flags.DEFINE_string(
    'reserved_cols', '',
    'columns to keep from input table,  they are separated with ,')
tf.app.flags.DEFINE_string(
    'output_cols', None,
    'output columns, such as: score float. multiple columns are separated by ,')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'predict batch size')
tf.app.flags.DEFINE_string(
    'profiling_file', None,
    'time stat file which can be viewed using chrome tracing')
tf.app.flags.DEFINE_string('redis_url', None, 'export to redis url, host:port')
tf.app.flags.DEFINE_string('redis_passwd', None, 'export to redis passwd')
tf.app.flags.DEFINE_integer('redis_threads', 5, 'export to redis threads')
tf.app.flags.DEFINE_integer('redis_batch_size', 1024,
                            'export to redis batch_size')
tf.app.flags.DEFINE_integer('redis_timeout', 600,
                            'export to redis time_out in seconds')
tf.app.flags.DEFINE_integer('redis_expire', 24,
                            'export to redis expire time in hour')
tf.app.flags.DEFINE_string('redis_embedding_version', '',
                           'redis embedding version')
tf.app.flags.DEFINE_integer('redis_write_kv', 1, 'whether write kv ')

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

tf.app.flags.DEFINE_bool('verbose', False, 'print more debug information')

# for automl hyper parameter tuning
tf.app.flags.DEFINE_string('model_dir', None, 'model directory')
tf.app.flags.DEFINE_bool('clear_model', False,
                         'remove model directory if exists')
tf.app.flags.DEFINE_string('hpo_param_path', None,
                           'hyperparameter tuning param path')
tf.app.flags.DEFINE_string('hpo_metric_save_path', None,
                           'hyperparameter save metric path')
tf.app.flags.DEFINE_string('asset_files', None, 'extra files to add to export')
tf.app.flags.DEFINE_bool('check_mode', False, 'is use check mode')
tf.app.flags.DEFINE_string('fg_json_path', None, '')

FLAGS = tf.app.flags.FLAGS


def check_param(name):
  assert getattr(FLAGS, name) != '', '%s should not be empty' % name


def set_selected_cols(pipeline_config, selected_cols, all_cols, all_col_types):
  if selected_cols:
    pipeline_config.data_config.selected_cols = selected_cols
    # add column types which will be used by OdpsInput, OdpsInputV2
    # to check consistency with input_fields.input_type
    if all_cols:
      all_cols_arr = all_cols.split(',')
      all_col_types_arr = all_col_types.split(',')
      all_col_types_map = {
          x.strip(): y.strip() for x, y in zip(all_cols_arr, all_col_types_arr)
      }
      selected_cols_arr = [x.strip() for x in selected_cols.split(',')]
      selected_col_types = [all_col_types_map[x] for x in selected_cols_arr]
      selected_col_types = ','.join(selected_col_types)
      pipeline_config.data_config.selected_col_types = selected_col_types

  print('[run.py] data_config.selected_cols = "%s"' %
        pipeline_config.data_config.selected_cols)
  print('[run.py] data_config.selected_col_types = "%s"' %
        pipeline_config.data_config.selected_col_types)


def main(argv):
  pai_util.set_on_pai()
  if FLAGS.distribute_eval:
    os.environ['distribute_eval'] = 'True'

  # load lookup op
  try:
    lookup_op_path = os.path.join(easy_rec.ops_dir, 'libembed_op.so')
    tf.load_op_library(lookup_op_path)
  except Exception as ex:
    print('Error: exception: %s' % str(ex))

  num_gpus_per_worker = FLAGS.num_gpus_per_worker
  worker_hosts = FLAGS.worker_hosts.split(',')
  num_worker = len(worker_hosts)
  assert FLAGS.distribute_strategy in DistributionStrategyMap, \
      'invalid distribute_strategy [%s], available ones are %s' % (
          FLAGS.distribute_strategy, ','.join(DistributionStrategyMap.keys()))

  if FLAGS.config:
    config = pai_util.process_config(FLAGS.config, FLAGS.task_index,
                                     len(FLAGS.worker_hosts.split(',')))
    pipeline_config = config_util.get_configs_from_pipeline_file(config, False)

    # should be in front of edit_config_json step
    # otherwise data_config and feature_config are not ready
    if pipeline_config.fg_json_path:
      fg_util.load_fg_json_to_config(pipeline_config)

  # if FLAGS.edit_config_json:
  #   print('[run.py] edit_config_json = %s' % FLAGS.edit_config_json)
  #   config_json = yaml.safe_load(FLAGS.edit_config_json)
  #   config_util.edit_config(pipeline_config, config_json)

  if FLAGS.model_dir:
    pipeline_config.model_dir = FLAGS.model_dir
    pipeline_config.model_dir = pipeline_config.model_dir.strip()
    print('[run.py] update model_dir to %s' % pipeline_config.model_dir)
    assert pipeline_config.model_dir.startswith(
        'oss://'), 'invalid model_dir format: %s' % pipeline_config.model_dir

  if FLAGS.config:
    if not pipeline_config.model_dir.endswith('/'):
      pipeline_config.model_dir += '/'

  if FLAGS.clear_model:
    if gfile.IsDirectory(pipeline_config.model_dir):
      gfile.DeleteRecursively(pipeline_config.model_dir)

  if FLAGS.cmd == 'train':
    assert FLAGS.config, 'config should not be empty when training!'

    if not FLAGS.train_tables:
      tables = FLAGS.tables.split(',')
      assert len(
          tables
      ) >= 2, 'at least 2 tables must be specified, but only[%d]: %s' % (
          len(tables), FLAGS.tables)

    if FLAGS.train_tables:
      pipeline_config.train_input_path = FLAGS.train_tables
    else:
      pipeline_config.train_input_path = FLAGS.tables.split(',')[0]

    if FLAGS.eval_tables:
      pipeline_config.eval_input_path = FLAGS.eval_tables
    else:
      pipeline_config.eval_input_path = FLAGS.tables.split(',')[1]

    print('[run.py] train_tables: %s' % pipeline_config.train_input_path)
    print('[run.py] eval_tables: %s' % pipeline_config.eval_input_path)

    if FLAGS.edit_config_json:
      print('[run.py] edit_config_json = %s' % FLAGS.edit_config_json)
      config_json = yaml.safe_load(FLAGS.edit_config_json)
      config_util.edit_config(pipeline_config, config_json)
    logging.info('edit json complete')
    logging.info(pipeline_config)

    if FLAGS.fine_tune_checkpoint:
      pipeline_config.train_config.fine_tune_checkpoint = FLAGS.fine_tune_checkpoint

    if FLAGS.boundary_table:
      logging.info('Load boundary_table: %s' % FLAGS.boundary_table)
      config_util.add_boundaries_to_config(pipeline_config,
                                           FLAGS.boundary_table)

    if FLAGS.sampler_table:
      pipeline_config.data_config.negative_sampler.input_path = FLAGS.sampler_table

    # parse selected_cols
    set_selected_cols(pipeline_config, FLAGS.selected_cols, FLAGS.all_cols,
                      FLAGS.all_col_types)

    distribute_strategy = DistributionStrategyMap[FLAGS.distribute_strategy]

    # update params specified by automl if hpo_param_path is specified
    if FLAGS.hpo_param_path:
      logging.info('hpo_param_path = %s' % FLAGS.hpo_param_path)
      with tf.gfile.GFile(FLAGS.hpo_param_path, 'r') as fin:
        hpo_config = yaml.safe_load(fin)
        hpo_params = hpo_config['param']
        config_util.edit_config(pipeline_config, hpo_params)
    config_util.auto_expand_share_feature_configs(pipeline_config)

    print('[run.py] with_evaluator %s' % str(FLAGS.with_evaluator))
    print('[run.py] eval_method %s' % FLAGS.eval_method)
    assert FLAGS.eval_method in [
        'none', 'master', 'separate'
    ], 'invalid evalaute_method: %s' % FLAGS.eval_method

    # with_evaluator is depreciated, keeped for compatibility
    if FLAGS.with_evaluator:
      FLAGS.eval_method = 'separate'

    num_worker = set_tf_config_and_get_train_worker_num(
        FLAGS.ps_hosts,
        FLAGS.worker_hosts,
        FLAGS.task_index,
        FLAGS.job_name,
        distribute_strategy=distribute_strategy,
        eval_method=FLAGS.eval_method)
    set_distribution_config(pipeline_config, num_worker, num_gpus_per_worker,
                            distribute_strategy)
    logging.info('run.py check_mode: %s .' % FLAGS.check_mode)
    train_and_evaluate_impl(
        pipeline_config,
        continue_train=FLAGS.continue_train,
        check_mode=FLAGS.check_mode)

    if FLAGS.hpo_metric_save_path:
      hpo_util.save_eval_metrics(
          pipeline_config.model_dir,
          metric_save_path=FLAGS.hpo_metric_save_path,
          has_evaluator=(FLAGS.eval_method == 'separate'))

  elif FLAGS.cmd == 'evaluate':
    check_param('config')
    # TODO: support multi-worker evaluation
    if not FLAGS.distribute_eval:
      assert len(
          FLAGS.worker_hosts.split(',')) == 1, 'evaluate only need 1 woker'
    config_util.auto_expand_share_feature_configs(pipeline_config)

    if FLAGS.eval_tables:
      pipeline_config.eval_input_path = FLAGS.eval_tables
    else:
      pipeline_config.eval_input_path = FLAGS.tables.split(',')[0]

    distribute_strategy = DistributionStrategyMap[FLAGS.distribute_strategy]
    set_tf_config_and_get_train_worker_num(
        FLAGS.ps_hosts,
        FLAGS.worker_hosts,
        FLAGS.task_index,
        FLAGS.job_name,
        eval_method='none')
    set_distribution_config(pipeline_config, num_worker, num_gpus_per_worker,
                            distribute_strategy)

    # parse selected_cols
    set_selected_cols(pipeline_config, FLAGS.selected_cols, FLAGS.all_cols,
                      FLAGS.all_col_types)
    if FLAGS.distribute_eval:
      os.environ['distribute_eval'] = 'True'
      logging.info('will_use_distribute_eval')
      distribute_eval = os.environ.get('distribute_eval')
      logging.info('distribute_eval = {}'.format(distribute_eval))
      easy_rec.distribute_evaluate(pipeline_config, FLAGS.checkpoint_path, None,
                                   FLAGS.eval_result_path)
    else:
      os.environ['distribute_eval'] = 'False'
      logging.info('will_use_eval')
      distribute_eval = os.environ.get('distribute_eval')
      logging.info('distribute_eval = {}'.format(distribute_eval))
      easy_rec.evaluate(pipeline_config, FLAGS.checkpoint_path, None,
                        FLAGS.eval_result_path)
  elif FLAGS.cmd == 'export':
    check_param('export_dir')
    check_param('config')

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
    if FLAGS.redis_write_kv:
      redis_params['redis_write_kv'] = FLAGS.redis_write_kv

    oss_params = {}
    if FLAGS.oss_path:
      oss_params['oss_path'] = FLAGS.oss_path
    if FLAGS.oss_endpoint:
      oss_params['oss_endpoint'] = FLAGS.oss_endpoint
    if FLAGS.oss_ak:
      oss_params['oss_ak'] = FLAGS.oss_ak
    if FLAGS.oss_sk:
      oss_params['oss_sk'] = FLAGS.oss_sk
    if FLAGS.oss_timeout > 0:
      oss_params['oss_timeout'] = FLAGS.oss_timeout
    if FLAGS.oss_expire > 0:
      oss_params['oss_expire'] = FLAGS.oss_expire
    if FLAGS.oss_threads > 0:
      oss_params['oss_threads'] = FLAGS.oss_threads
    if FLAGS.oss_embedding_version:
      redis_params['oss_embedding_version'] = FLAGS.oss_embedding_version
    if FLAGS.oss_write_kv:
      oss_params['oss_write_kv'] = True if FLAGS.oss_write_kv == 1 else False

    set_tf_config_and_get_train_worker_num(
        FLAGS.ps_hosts,
        FLAGS.worker_hosts,
        FLAGS.task_index,
        FLAGS.job_name,
        eval_method='none')

    assert len(FLAGS.worker_hosts.split(',')) == 1, 'export only need 1 woker'
    config_util.auto_expand_share_feature_configs(pipeline_config)

    export_dir = FLAGS.export_dir
    if not export_dir.endswith('/'):
      export_dir = export_dir + '/'
    if FLAGS.clear_export:
      if gfile.IsDirectory(export_dir):
        gfile.DeleteRecursively(export_dir)

    extra_params = redis_params
    extra_params.update(oss_params)
    easy_rec.export(export_dir, pipeline_config, FLAGS.checkpoint_path,
                    FLAGS.asset_files, FLAGS.verbose, **extra_params)
  elif FLAGS.cmd == 'predict':
    check_param('tables')
    check_param('saved_model_dir')
    logging.info('will use the following columns as model input: %s' %
                 FLAGS.selected_cols)
    logging.info('will copy the following columns to output: %s' %
                 FLAGS.reserved_cols)

    profiling_file = FLAGS.profiling_file if FLAGS.task_index == 0 else None
    if profiling_file is not None:
      print('profiling_file = %s ' % profiling_file)
    predictor = ODPSPredictor(
        FLAGS.saved_model_dir,
        fg_json_path=FLAGS.fg_json_path,
        profiling_file=profiling_file,
        all_cols=FLAGS.all_cols,
        all_col_types=FLAGS.all_col_types)
    input_table, output_table = FLAGS.tables, FLAGS.outputs
    logging.info('input_table = %s, output_table = %s' %
                 (input_table, output_table))
    worker_num = len(FLAGS.worker_hosts.split(','))
    predictor.predict_impl(
        input_table,
        output_table,
        reserved_cols=FLAGS.reserved_cols,
        output_cols=FLAGS.output_cols,
        batch_size=FLAGS.batch_size,
        slice_id=FLAGS.task_index,
        slice_num=worker_num)
  elif FLAGS.cmd == 'export_checkpoint':
    check_param('export_dir')
    check_param('config')
    set_tf_config_and_get_train_worker_num(
        FLAGS.ps_hosts,
        FLAGS.worker_hosts,
        FLAGS.task_index,
        FLAGS.job_name,
        eval_method='none')
    assert len(FLAGS.worker_hosts.split(',')) == 1, 'export only need 1 woker'
    config_util.auto_expand_share_feature_configs(pipeline_config)
    easy_rec.export_checkpoint(
        pipeline_config,
        export_path=FLAGS.export_dir + '/model',
        checkpoint_path=FLAGS.checkpoint_path,
        asset_files=FLAGS.asset_files,
        verbose=FLAGS.verbose)
  elif FLAGS.cmd == 'vector_retrieve':
    check_param('knn_distance')
    assert FLAGS.knn_feature_dims is not None, '`knn_feature_dims` should not be None'
    assert FLAGS.knn_num_neighbours is not None, '`knn_num_neighbours` should not be None'

    query_table, doc_table, output_table = FLAGS.query_table, FLAGS.doc_table, FLAGS.outputs
    if not query_table:
      tables = FLAGS.tables.split(',')
      assert len(
          tables
      ) >= 1, 'at least 1 tables must be specified, but only[%d]: %s' % (
          len(tables), FLAGS.tables)
      query_table = tables[0]
      doc_table = tables[1] if len(tables) > 1 else query_table

    knn = VectorRetrieve(
        query_table,
        doc_table,
        output_table,
        ndim=FLAGS.knn_feature_dims,
        distance=1 if FLAGS.knn_distance == 'inner_product' else 0,
        delimiter=FLAGS.knn_feature_delimiter,
        batch_size=FLAGS.batch_size,
        index_type=FLAGS.knn_index_type,
        nlist=FLAGS.knn_nlist,
        nprobe=FLAGS.knn_nprobe,
        m=FLAGS.knn_compress_dim)
    worker_hosts = FLAGS.worker_hosts.split(',')
    knn(FLAGS.knn_num_neighbours, FLAGS.task_index, len(worker_hosts))
  elif FLAGS.cmd == 'check':
    run_check(pipeline_config, FLAGS.tables)
  else:
    raise ValueError(
        'cmd should be one of train/evaluate/export/predict/export_checkpoint/vector_retrieve'
    )


if __name__ == '__main__':
  tf.app.run()
