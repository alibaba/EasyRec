# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import math
import os
import time

import six
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.platform import gfile

import easy_rec
from easy_rec.python.builders import strategy_builder
from easy_rec.python.compat import estimator_train
from easy_rec.python.compat import exporter
from easy_rec.python.input.input import Input
from easy_rec.python.model.easy_rec_estimator import EasyRecEstimator
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.train_pb2 import DistributionStrategy
from easy_rec.python.utils import config_util
from easy_rec.python.utils import constant
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import fg_util
from easy_rec.python.utils import load_class
from easy_rec.python.utils.config_util import get_eval_input_path
from easy_rec.python.utils.config_util import get_model_dir_path
from easy_rec.python.utils.config_util import get_train_input_path
from easy_rec.python.utils.config_util import set_eval_input_path
from easy_rec.python.utils.export_big_model import export_big_model
from easy_rec.python.utils.export_big_model import export_big_model_to_oss

try:
  import horovod.tensorflow as hvd
except Exception:
  hvd = None

if tf.__version__ >= '2.0':
  from tensorflow.core.protobuf import config_pb2

  ConfigProto = config_pb2.ConfigProto
  GPUOptions = config_pb2.GPUOptions

  tf = tf.compat.v1
else:
  GPUOptions = tf.GPUOptions
  ConfigProto = tf.ConfigProto

load_class.auto_import()

# when version of tensorflow > 1.8 strip_default_attrs set true will cause
# saved_model inference core, such as:
#   [libprotobuf FATAL external/protobuf_archive/src/google/protobuf/map.h:1058]
#    CHECK failed: it != end(): key not found: new_axis_mask
# so temporarily modify strip_default_attrs of _SavedModelExporter in
# tf.estimator.exporter to false by default
FinalExporter = exporter.FinalExporter
LatestExporter = exporter.LatestExporter
BestExporter = exporter.BestExporter


def _get_input_fn(data_config,
                  feature_configs,
                  data_path=None,
                  export_config=None,
                  check_mode=False,
                  **kwargs):
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

  task_id, task_num = estimator_utils.get_task_index_and_num()
  input_obj = input_class(
      data_config,
      feature_configs,
      data_path,
      task_index=task_id,
      task_num=task_num,
      check_mode=check_mode,
      **kwargs)
  input_fn = input_obj.create_input(export_config)
  return input_fn


def _create_estimator(pipeline_config, distribution=None, params={}):
  model_config = pipeline_config.model_config
  train_config = pipeline_config.train_config
  gpu_options = GPUOptions(allow_growth=True)  # False)

  logging.info(
      'train_config.train_distribute=%s[value=%d]' %
      (DistributionStrategy.Name(pipeline_config.train_config.train_distribute),
       pipeline_config.train_config.train_distribute))

  # set gpu options only under hvd scenes
  if hvd is not None and pipeline_config.train_config.train_distribute in [
      DistributionStrategy.EmbeddingParallelStrategy,
      DistributionStrategy.SokStrategy, DistributionStrategy.HorovodStrategy
  ]:
    local_rnk = hvd.local_rank()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logging.info('local_rnk=%d num_gpus=%d' % (local_rnk, len(gpus)))
    if len(gpus) > 0:
      tf.config.experimental.set_visible_devices(gpus[local_rnk], 'GPU')
      gpu_options.visible_device_list = str(local_rnk)

  session_config = ConfigProto(
      gpu_options=gpu_options,
      allow_soft_placement=True,
      log_device_placement=params.get('log_device_placement', False),
      inter_op_parallelism_threads=train_config.inter_op_parallelism_threads,
      intra_op_parallelism_threads=train_config.intra_op_parallelism_threads)

  if constant.NO_ARITHMETRIC_OPTI in os.environ:
    logging.info('arithmetic_optimization is closed to improve performance')
    session_config.graph_options.rewrite_options.arithmetic_optimization = \
        session_config.graph_options.rewrite_options.OFF

  session_config.device_filters.append('/job:ps')
  model_cls = EasyRecModel.create_class(model_config.model_class)

  save_checkpoints_steps = None
  save_checkpoints_secs = None
  if train_config.HasField('save_checkpoints_steps'):
    save_checkpoints_steps = train_config.save_checkpoints_steps
  if train_config.HasField('save_checkpoints_secs'):
    save_checkpoints_secs = train_config.save_checkpoints_secs
  # if both `save_checkpoints_steps` and `save_checkpoints_secs` are not set,
  # use the default value of save_checkpoints_steps
  if save_checkpoints_steps is None and save_checkpoints_secs is None:
    save_checkpoints_steps = train_config.save_checkpoints_steps

  run_config = tf.estimator.RunConfig(
      model_dir=pipeline_config.model_dir,
      log_step_count_steps=None,  # train_config.log_step_count_steps,
      save_summary_steps=train_config.save_summary_steps,
      save_checkpoints_steps=save_checkpoints_steps,
      save_checkpoints_secs=save_checkpoints_secs,
      keep_checkpoint_max=train_config.keep_checkpoint_max,
      train_distribute=distribution,
      eval_distribute=distribution,
      session_config=session_config)

  estimator = EasyRecEstimator(
      pipeline_config, model_cls, run_config=run_config, params=params)
  return estimator, run_config


def _create_eval_export_spec(pipeline_config, eval_data, check_mode=False):
  data_config = pipeline_config.data_config
  # feature_configs = pipeline_config.feature_configs
  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
  eval_config = pipeline_config.eval_config
  export_config = pipeline_config.export_config
  if eval_config.num_examples > 0:
    eval_steps = int(
        math.ceil(float(eval_config.num_examples) / data_config.batch_size))
    logging.info('eval_steps = %d' % eval_steps)
  else:
    eval_steps = None
  input_fn_kwargs = {'pipeline_config': pipeline_config}
  if data_config.input_type == data_config.InputType.OdpsRTPInputV2:
    input_fn_kwargs['fg_json_path'] = pipeline_config.fg_json_path
  # create eval input
  export_input_fn = _get_input_fn(
      data_config,
      feature_configs,
      None,
      export_config,
      check_mode=check_mode,
      **input_fn_kwargs)
  if export_config.exporter_type == 'final':
    exporters = [
        FinalExporter(name='final', serving_input_receiver_fn=export_input_fn)
    ]
  elif export_config.exporter_type == 'latest':
    exporters = [
        LatestExporter(
            name='latest',
            serving_input_receiver_fn=export_input_fn,
            exports_to_keep=export_config.exports_to_keep)
    ]
  elif export_config.exporter_type == 'best':
    logging.info(
        'will use BestExporter, metric is %s, the bigger the better: %d' %
        (export_config.best_exporter_metric, export_config.metric_bigger))

    def _metric_cmp_fn(best_eval_result, current_eval_result):
      logging.info('metric: best = %s current = %s' %
                   (str(best_eval_result), str(current_eval_result)))
      if export_config.metric_bigger:
        return (best_eval_result[export_config.best_exporter_metric] <
                current_eval_result[export_config.best_exporter_metric])
      else:
        return (best_eval_result[export_config.best_exporter_metric] >
                current_eval_result[export_config.best_exporter_metric])

    exporters = [
        BestExporter(
            name='best',
            serving_input_receiver_fn=export_input_fn,
            compare_fn=_metric_cmp_fn,
            exports_to_keep=export_config.exports_to_keep)
    ]
  elif export_config.exporter_type == 'none':
    exporters = []
  else:
    raise ValueError('Unknown exporter type %s' % export_config.exporter_type)

  # set throttle_secs to a small number, so that we can control evaluation
  # interval steps by checkpoint saving steps
  eval_input_fn = _get_input_fn(data_config, feature_configs, eval_data,
                                **input_fn_kwargs)
  eval_spec = tf.estimator.EvalSpec(
      name='val',
      input_fn=eval_input_fn,
      steps=eval_steps,
      throttle_secs=10,
      exporters=exporters)
  return eval_spec


def _check_model_dir(model_dir, continue_train):
  if not continue_train:
    if not gfile.IsDirectory(model_dir):
      gfile.MakeDirs(model_dir)
    else:
      assert len(gfile.Glob(model_dir + '/model.ckpt-*.meta')) == 0, \
          'model_dir[=%s] already exists and not empty(if you ' \
          'want to continue train on current model_dir please ' \
          'delete dir %s or specify --continue_train[internal use only])' % (
              model_dir, model_dir)
  else:
    if not gfile.IsDirectory(model_dir):
      logging.info('%s does not exists, create it automatically' % model_dir)
      gfile.MakeDirs(model_dir)


def _get_ckpt_path(pipeline_config, checkpoint_path):
  if checkpoint_path != '' and checkpoint_path is not None:
    if gfile.IsDirectory(checkpoint_path):
      ckpt_path = estimator_utils.latest_checkpoint(checkpoint_path)
    else:
      ckpt_path = checkpoint_path
  elif gfile.IsDirectory(pipeline_config.model_dir):
    ckpt_path = estimator_utils.latest_checkpoint(pipeline_config.model_dir)
    logging.info('checkpoint_path is not specified, '
                 'will use latest checkpoint %s from %s' %
                 (ckpt_path, pipeline_config.model_dir))
  else:
    assert False, 'pipeline_config.model_dir(%s) does not exist' \
                  % pipeline_config.model_dir
  return ckpt_path


def train_and_evaluate(pipeline_config_path, continue_train=False):
  """Train and evaluate a EasyRec model defined in pipeline_config_path.

  Build an EasyRecEstimator, and then train and evaluate the estimator.

  Args:
    pipeline_config_path: a path to EasyRecConfig object, specifies
    train_config: model_config, data_config and eval_config
    continue_train: whether to restart train from an existing
                    checkpoint
  Returns:
    None, the model will be saved into pipeline_config.model_dir
  """
  assert gfile.Exists(pipeline_config_path), 'pipeline_config_path not exists'
  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)

  _train_and_evaluate_impl(pipeline_config, continue_train)

  return pipeline_config


def _train_and_evaluate_impl(pipeline_config,
                             continue_train=False,
                             check_mode=False,
                             fit_on_eval=False,
                             fit_on_eval_steps=None):
  train_config = pipeline_config.train_config
  data_config = pipeline_config.data_config
  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)

  if train_config.train_distribute != DistributionStrategy.NoStrategy\
      and train_config.sync_replicas:
    logging.warning(
        'will set sync_replicas to False, because train_distribute[%s] != NoStrategy'
        % pipeline_config.train_config.train_distribute)
    pipeline_config.train_config.sync_replicas = False

  train_data = get_train_input_path(pipeline_config)
  eval_data = get_eval_input_path(pipeline_config)

  distribution = strategy_builder.build(train_config)
  params = {}
  if train_config.is_profiling:
    params['log_device_placement'] = True
  estimator, run_config = _create_estimator(
      pipeline_config, distribution=distribution, params=params)

  version_file = os.path.join(pipeline_config.model_dir, 'version')
  if estimator_utils.is_chief():
    _check_model_dir(pipeline_config.model_dir, continue_train)
    config_util.save_pipeline_config(pipeline_config, pipeline_config.model_dir)
    with gfile.GFile(version_file, 'w') as f:
      f.write(easy_rec.__version__ + '\n')

  train_steps = None
  if train_config.HasField('num_steps') and train_config.num_steps > 0:
    train_steps = train_config.num_steps
  assert train_steps is not None or data_config.num_epochs > 0, (
      'either num_steps and num_epochs must be set to an integer > 0.')

  if train_steps and data_config.num_epochs:
    logging.info('Both num_steps and num_epochs are set.')
    is_sync = train_config.sync_replicas
    batch_size = data_config.batch_size
    epoch_str = 'sample_num * %d / %d' % (data_config.num_epochs, batch_size)
    if is_sync:
      _, worker_num = estimator_utils.get_task_index_and_num()
      epoch_str += ' / ' + str(worker_num)
    logging.info('Will train min(%d, %s) steps...' % (train_steps, epoch_str))

  input_fn_kwargs = {'pipeline_config': pipeline_config}
  if data_config.input_type == data_config.InputType.OdpsRTPInputV2:
    input_fn_kwargs['fg_json_path'] = pipeline_config.fg_json_path

  # create train input
  train_input_fn = _get_input_fn(
      data_config,
      feature_configs,
      train_data,
      check_mode=check_mode,
      **input_fn_kwargs)
  # Currently only a single Eval Spec is allowed.
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=train_steps)

  embedding_parallel = train_config.train_distribute in (
      DistributionStrategy.SokStrategy,
      DistributionStrategy.EmbeddingParallelStrategy)

  if embedding_parallel:
    estimator.train(
        input_fn=train_input_fn,
        max_steps=train_spec.max_steps,
        hooks=list(train_spec.hooks),
        saving_listeners=train_spec.saving_listeners)
    train_input_fn.input_creator.stop()
  else:
    # create eval spec
    eval_spec = _create_eval_export_spec(
        pipeline_config, eval_data, check_mode=check_mode)
    estimator_train.train_and_evaluate(estimator, train_spec, eval_spec)
  logging.info('Train and evaluate finish')
  if fit_on_eval and (not estimator_utils.is_evaluator()):
    tf.reset_default_graph()
    logging.info('Start continue training on eval data')
    eval_input_fn = _get_input_fn(data_config, feature_configs, eval_data,
                                  **input_fn_kwargs)
    if fit_on_eval_steps is not None:
      # wait estimator train done to get the correct train_steps
      while not estimator_train.estimator_train_done(estimator):
        time.sleep(1)
      train_steps = estimator_utils.get_trained_steps(estimator.model_dir)
      logging.info('\ttrain_steps=%d fit_on_eval_steps=%d' %
                   (train_steps, fit_on_eval_steps))
      fit_on_eval_steps += train_steps
    # Do not use estimator_train.train_and_evaluate as it starts tf.Server,
    # which is redundant and reports port not available error.
    estimator.train(
        input_fn=eval_input_fn,
        max_steps=fit_on_eval_steps,
        hooks=list(train_spec.hooks),
        saving_listeners=train_spec.saving_listeners if hasattr(
            train_spec, 'saving_listeners') else None)
    logging.info('Finished training on eval data')
  # return estimator for custom training using estimator.train
  return estimator


def evaluate(pipeline_config,
             eval_checkpoint_path='',
             eval_data_path=None,
             eval_result_filename='eval_result.txt'):
  """Evaluate a EasyRec model defined in pipeline_config_path.

  Evaluate the model defined in pipeline_config_path on the eval data,
  the metrics will be displayed on tensorboard and saved into eval_result.txt.

  Args:
    pipeline_config: either EasyRecConfig path or its instance
    eval_checkpoint_path: if specified, will use this model instead of
        model specified by model_dir in pipeline_config_path
    eval_data_path: eval data path, default use eval data in pipeline_config
        could be a path or a list of paths
    eval_result_filename: evaluation result metrics save path.

  Returns:
    A dict of evaluation metrics: the metrics are specified in
        pipeline_config_path
    global_step: the global step for which this evaluation was performed.

  Raises:
    AssertionError, if:
      * pipeline_config_path does not exist
  """
  pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config)
  if pipeline_config.fg_json_path:
    fg_util.load_fg_json_to_config(pipeline_config)
  if eval_data_path is not None:
    logging.info('Evaluating on data: %s' % eval_data_path)
    set_eval_input_path(pipeline_config, eval_data_path)

  train_config = pipeline_config.train_config
  eval_data = get_eval_input_path(pipeline_config)

  server_target = None
  if 'TF_CONFIG' in os.environ:
    tf_config = estimator_utils.chief_to_master()
    from tensorflow.python.training import server_lib
    if tf_config['task']['type'] == 'ps':
      cluster = tf.train.ClusterSpec(tf_config['cluster'])
      server = server_lib.Server(
          cluster, job_name='ps', task_index=tf_config['task']['index'])
      server.join()
    elif tf_config['task']['type'] == 'master':
      if 'ps' in tf_config['cluster']:
        cluster = tf.train.ClusterSpec(tf_config['cluster'])
        server = server_lib.Server(cluster, job_name='master', task_index=0)
        server_target = server.target
        print('server_target = %s' % server_target)

  distribution = strategy_builder.build(train_config)
  estimator, run_config = _create_estimator(pipeline_config, distribution)
  eval_spec = _create_eval_export_spec(pipeline_config, eval_data)
  ckpt_path = _get_ckpt_path(pipeline_config, eval_checkpoint_path)

  if server_target:
    # evaluate with parameter server
    input_iter = eval_spec.input_fn(
        mode=tf.estimator.ModeKeys.EVAL).make_one_shot_iterator()
    input_feas, input_lbls = input_iter.get_next()
    from tensorflow.python.training.device_setter import replica_device_setter
    from tensorflow.python.framework.ops import device
    from tensorflow.python.training.monitored_session import MonitoredSession
    from tensorflow.python.training.monitored_session import ChiefSessionCreator
    with device(
        replica_device_setter(
            worker_device='/job:master/task:0', cluster=cluster)):
      estimator_spec = estimator._eval_model_fn(input_feas, input_lbls,
                                                run_config)

    session_config = ConfigProto(
        allow_soft_placement=True, log_device_placement=True)
    chief_sess_creator = ChiefSessionCreator(
        master=server_target,
        checkpoint_filename_with_path=ckpt_path,
        config=session_config)
    eval_metric_ops = estimator_spec.eval_metric_ops
    update_ops = [eval_metric_ops[x][1] for x in eval_metric_ops.keys()]
    metric_ops = {x: eval_metric_ops[x][0] for x in eval_metric_ops.keys()}
    update_op = tf.group(update_ops)
    with MonitoredSession(
        session_creator=chief_sess_creator,
        hooks=None,
        stop_grace_period_secs=120) as sess:
      while True:
        try:
          sess.run(update_op)
        except tf.errors.OutOfRangeError:
          break
      eval_result = sess.run(metric_ops)
  else:
    # this way does not work, wait to be debugged
    # the variables are not placed to parameter server
    # with tf.device(
    #    replica_device_setter(
    #        worker_device='/job:master/task:0', cluster=cluster)):
    eval_result = estimator.evaluate(
        eval_spec.input_fn, eval_spec.steps, checkpoint_path=ckpt_path)
  eval_spec.input_fn.input_creator.stop()
  logging.info('Evaluate finish')

  print('eval_result = ', eval_result)
  logging.info('eval_result = {0}'.format(eval_result))
  # write eval result to file
  model_dir = pipeline_config.model_dir
  eval_result_file = os.path.join(model_dir, eval_result_filename)
  logging.info('save eval result to file %s' % eval_result_file)
  with gfile.GFile(eval_result_file, 'w') as ofile:
    result_to_write = {}
    for key in sorted(eval_result):
      # skip logging binary data
      if isinstance(eval_result[key], six.binary_type):
        continue
      # convert numpy float to python float
      result_to_write[key] = eval_result[key].item()
    ofile.write(json.dumps(result_to_write, indent=2))
  return eval_result


def distribute_evaluate(pipeline_config,
                        eval_checkpoint_path='',
                        eval_data_path=None,
                        eval_result_filename='distribute_eval_result.txt'):
  """Evaluate a EasyRec model defined in pipeline_config_path.

  Evaluate the model defined in pipeline_config_path on the eval data,
  the metrics will be displayed on tensorboard and saved into eval_result.txt.

  Args:
    pipeline_config: either EasyRecConfig path or its instance
    eval_checkpoint_path: if specified, will use this model instead of
        model specified by model_dir in pipeline_config_path
    eval_data_path: eval data path, default use eval data in pipeline_config
        could be a path or a list of paths
    eval_result_filename: evaluation result metrics save path.

  Returns:
    A dict of evaluation metrics: the metrics are specified in
        pipeline_config_path
    global_step: the global step for which this evaluation was performed.

  Raises:
    AssertionError, if:
      * pipeline_config_path does not exist
  """
  pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config)
  if eval_data_path is not None:
    logging.info('Evaluating on data: %s' % eval_data_path)
    set_eval_input_path(pipeline_config, eval_data_path)
  train_config = pipeline_config.train_config
  eval_data = get_eval_input_path(pipeline_config)
  data_config = pipeline_config.data_config
  if data_config.HasField('sampler'):
    logging.warning(
        'It is not accuracy to use eval with negative sampler, recommand to use hitrate.py!'
    )
    eval_result = {}
    return eval_result
  model_dir = get_model_dir_path(pipeline_config)
  eval_tmp_results_dir = os.path.join(model_dir, 'distribute_eval_tmp_results')
  if not gfile.IsDirectory(eval_tmp_results_dir):
    logging.info('create eval tmp results dir {}'.format(eval_tmp_results_dir))
    gfile.MakeDirs(eval_tmp_results_dir)
  assert gfile.IsDirectory(
      eval_tmp_results_dir), 'tmp results dir not create success.'
  os.environ['eval_tmp_results_dir'] = eval_tmp_results_dir

  server_target = None
  cur_job_name = None
  if 'TF_CONFIG' in os.environ:
    tf_config = estimator_utils.chief_to_master()

    from tensorflow.python.training import server_lib
    if tf_config['task']['type'] == 'ps':
      cluster = tf.train.ClusterSpec(tf_config['cluster'])
      server = server_lib.Server(
          cluster, job_name='ps', task_index=tf_config['task']['index'])
      server.join()
    elif tf_config['task']['type'] == 'master':
      if 'ps' in tf_config['cluster']:
        cur_job_name = tf_config['task']['type']
        cur_task_index = tf_config['task']['index']
        cluster = tf.train.ClusterSpec(tf_config['cluster'])
        server = server_lib.Server(
            cluster, job_name=cur_job_name, task_index=cur_task_index)
        server_target = server.target
        print('server_target = %s' % server_target)
    elif tf_config['task']['type'] == 'worker':
      if 'ps' in tf_config['cluster']:
        cur_job_name = tf_config['task']['type']
        cur_task_index = tf_config['task']['index']
        cluster = tf.train.ClusterSpec(tf_config['cluster'])
        server = server_lib.Server(
            cluster, job_name=cur_job_name, task_index=cur_task_index)
        server_target = server.target
        print('server_target = %s' % server_target)

  if server_target:
    from tensorflow.python.training.device_setter import replica_device_setter
    from tensorflow.python.framework.ops import device
    from tensorflow.python.training.monitored_session import MonitoredSession
    from tensorflow.python.training.monitored_session import ChiefSessionCreator
    from tensorflow.python.training.monitored_session import WorkerSessionCreator
    from easy_rec.python.utils.estimator_utils import EvaluateExitBarrierHook
    cur_work_device = '/job:' + cur_job_name + '/task:' + str(cur_task_index)
    cur_ps_num = len(tf_config['cluster']['ps'])
    with device(
        replica_device_setter(
            ps_tasks=cur_ps_num, worker_device=cur_work_device,
            cluster=cluster)):
      distribution = strategy_builder.build(train_config)
      estimator, run_config = _create_estimator(pipeline_config, distribution)
      eval_spec = _create_eval_export_spec(pipeline_config, eval_data)
      ckpt_path = _get_ckpt_path(pipeline_config, eval_checkpoint_path)
      ckpt_dir = os.path.dirname(ckpt_path)
      input_iter = eval_spec.input_fn(
          mode=tf.estimator.ModeKeys.EVAL).make_one_shot_iterator()
      input_feas, input_lbls = input_iter.get_next()
      estimator_spec = estimator._distribute_eval_model_fn(
          input_feas, input_lbls, run_config)

    session_config = ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        device_filters=['/job:ps',
                        '/job:worker/task:%d' % cur_task_index])
    if cur_job_name == 'master':
      metric_variables = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
      model_ready_for_local_init_op = tf.variables_initializer(metric_variables)
      global_variables = tf.global_variables()
      remain_variables = list(
          set(global_variables).difference(set(metric_variables)))
      cur_saver = tf.train.Saver(var_list=remain_variables, sharded=True)
      cur_scaffold = tf.train.Scaffold(
          saver=cur_saver,
          ready_for_local_init_op=model_ready_for_local_init_op)
      cur_sess_creator = ChiefSessionCreator(
          scaffold=cur_scaffold,
          master=server_target,
          checkpoint_filename_with_path=ckpt_path,
          config=session_config)
    else:
      cur_sess_creator = WorkerSessionCreator(
          master=server_target, config=session_config)
    eval_metric_ops = estimator_spec.eval_metric_ops
    update_ops = [eval_metric_ops[x][1] for x in eval_metric_ops.keys()]
    metric_ops = {x: eval_metric_ops[x][0] for x in eval_metric_ops.keys()}
    update_op = tf.group(update_ops)
    cur_worker_num = len(tf_config['cluster']['worker']) + 1
    if cur_job_name == 'master':
      cur_stop_grace_period_sesc = 120
      cur_hooks = EvaluateExitBarrierHook(cur_worker_num, True, ckpt_dir,
                                          metric_ops)
    else:
      cur_stop_grace_period_sesc = 10
      cur_hooks = EvaluateExitBarrierHook(cur_worker_num, False, ckpt_dir,
                                          metric_ops)
    with MonitoredSession(
        session_creator=cur_sess_creator,
        hooks=[cur_hooks],
        stop_grace_period_secs=cur_stop_grace_period_sesc) as sess:
      while True:
        try:
          sess.run(update_op)
        except tf.errors.OutOfRangeError:
          break
    eval_result = cur_hooks.eval_result

  logging.info('Evaluate finish')

  # write eval result to file
  model_dir = pipeline_config.model_dir
  eval_result_file = os.path.join(model_dir, eval_result_filename)
  logging.info('save eval result to file %s' % eval_result_file)
  if cur_job_name == 'master':
    print('eval_result = ', eval_result)
    logging.info('eval_result = {0}'.format(eval_result))
    with gfile.GFile(eval_result_file, 'w') as ofile:
      result_to_write = {'eval_method': 'distribute'}
      for key in sorted(eval_result):
        # skip logging binary data
        if isinstance(eval_result[key], six.binary_type):
          continue
        # convert numpy float to python float
        result_to_write[key] = eval_result[key].item()

      ofile.write(json.dumps(result_to_write))
  return eval_result


def predict(pipeline_config, checkpoint_path='', data_path=None):
  """Predict a EasyRec model defined in pipeline_config_path.

  Predict the model defined in pipeline_config_path on the eval data.

  Args:
    pipeline_config: either EasyRecConfig path or its instance
    checkpoint_path: if specified, will use this model instead of
        model specified by model_dir in pipeline_config_path
    data_path: data path, default use eval data in pipeline_config
        could be a path or a list of paths

  Returns:
    A list of dict of predict results

  Raises:
    AssertionError, if:
      * pipeline_config_path does not exist
  """
  pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config)
  if pipeline_config.fg_json_path:
    fg_util.load_fg_json_to_config(pipeline_config)
  if data_path is not None:
    logging.info('Predict on data: %s' % data_path)
    set_eval_input_path(pipeline_config, data_path)
  train_config = pipeline_config.train_config
  eval_data = get_eval_input_path(pipeline_config)

  distribution = strategy_builder.build(train_config)
  estimator, _ = _create_estimator(pipeline_config, distribution)
  eval_spec = _create_eval_export_spec(pipeline_config, eval_data)

  ckpt_path = _get_ckpt_path(pipeline_config, checkpoint_path)

  pred_result = estimator.predict(eval_spec.input_fn, checkpoint_path=ckpt_path)
  logging.info('Predict finish')
  return pred_result


def export(export_dir,
           pipeline_config,
           checkpoint_path='',
           asset_files=None,
           verbose=False,
           **extra_params):
  """Export model defined in pipeline_config_path.

  Args:
    export_dir: base directory where the model should be exported
    pipeline_config: proto.EasyRecConfig instance or file path
       specify proto.EasyRecConfig
    checkpoint_path: if specified, will use this model instead of
       model in model_dir in pipeline_config_path
    asset_files: extra files to add to assets, comma separated;
       if asset file variable in graph need to be renamed,
       specify by new_file_name:file_path
    version: if version is defined, then will skip writing embedding to redis,
       assume that embedding is already write into redis
    verbose: dumps debug information
    extra_params: keys related to write embedding to redis/oss
       redis_url, redis_passwd, redis_threads, redis_batch_size,
       redis_timeout, redis_expire if export embedding to redis;
       redis_embedding_version: if specified, will kill export to redis
       --
       oss_path, oss_endpoint, oss_ak, oss_sk, oss_timeout,
       oss_expire, oss_write_kv, oss_embedding_version

  Returns:
    the directory where model is exported

  Raises:
    AssertionError, if:
      * pipeline_config_path does not exist
  """
  if not gfile.Exists(export_dir):
    gfile.MakeDirs(export_dir)

  pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config)
  if pipeline_config.fg_json_path:
    fg_util.load_fg_json_to_config(pipeline_config)
  # feature_configs = pipeline_config.feature_configs
  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
  # create estimator
  params = {'log_device_placement': verbose}
  if asset_files:
    logging.info('will add asset files: %s' % asset_files)
    asset_file_dict = {}
    for asset_file in asset_files.split(','):
      asset_file = asset_file.strip()
      if ':' not in asset_file or asset_file.startswith(
          'oss:') or asset_file.startswith('hdfs:'):
        _, asset_name = os.path.split(asset_file)
      else:
        asset_name, asset_file = asset_file.split(':', 1)
      asset_file_dict[asset_name] = asset_file
    params['asset_files'] = asset_file_dict
  estimator, _ = _create_estimator(pipeline_config, params=params)
  # construct serving input fn
  export_config = pipeline_config.export_config
  data_config = pipeline_config.data_config
  input_fn_kwargs = {'pipeline_config': pipeline_config}
  if data_config.input_type == data_config.InputType.OdpsRTPInputV2:
    input_fn_kwargs['fg_json_path'] = pipeline_config.fg_json_path
  serving_input_fn = _get_input_fn(data_config, feature_configs, None,
                                   export_config, **input_fn_kwargs)
  ckpt_path = _get_ckpt_path(pipeline_config, checkpoint_path)
  if 'oss_path' in extra_params:
    if pipeline_config.train_config.HasField('incr_save_config'):
      incr_save_config = pipeline_config.train_config.incr_save_config
      extra_params['incr_update'] = {}
      incr_save_type = incr_save_config.WhichOneof('incr_update')
      logging.info('incr_save_type=%s' % incr_save_type)
      if incr_save_type:
        extra_params['incr_update'][incr_save_type] = getattr(
            incr_save_config, incr_save_type)
    return export_big_model_to_oss(export_dir, pipeline_config, extra_params,
                                   serving_input_fn, estimator, ckpt_path,
                                   verbose)

  if 'redis_url' in extra_params:
    return export_big_model(export_dir, pipeline_config, extra_params,
                            serving_input_fn, estimator, ckpt_path, verbose)

  final_export_dir = estimator.export_savedmodel(
      export_dir_base=export_dir,
      serving_input_receiver_fn=serving_input_fn,
      checkpoint_path=ckpt_path,
      strip_default_attrs=True)

  # add export ts as version info
  saved_model = saved_model_pb2.SavedModel()
  if type(final_export_dir) not in [type(''), type(u'')]:
    final_export_dir = final_export_dir.decode('utf-8')
  export_ts = [
      x for x in final_export_dir.split('/') if x != '' and x is not None
  ]
  export_ts = export_ts[-1]
  saved_pb_path = os.path.join(final_export_dir, 'saved_model.pb')
  with gfile.GFile(saved_pb_path, 'rb') as fin:
    saved_model.ParseFromString(fin.read())
  saved_model.meta_graphs[0].meta_info_def.meta_graph_version = export_ts
  with gfile.GFile(saved_pb_path, 'wb') as fout:
    fout.write(saved_model.SerializeToString())

  logging.info('model has been exported to %s successfully' % final_export_dir)
  return final_export_dir


def export_checkpoint(pipeline_config=None,
                      export_path='',
                      checkpoint_path='',
                      asset_files=None,
                      verbose=False,
                      mode=tf.estimator.ModeKeys.PREDICT):
  """Export the EasyRec model as checkpoint."""
  pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config)
  if pipeline_config.fg_json_path:
    fg_util.load_fg_json_to_config(pipeline_config)
  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
  data_config = pipeline_config.data_config

  input_fn_kwargs = {'pipeline_config': pipeline_config}
  if data_config.input_type == data_config.InputType.OdpsRTPInputV2:
    input_fn_kwargs['fg_json_path'] = pipeline_config.fg_json_path

  # create estimator
  params = {'log_device_placement': verbose}
  if asset_files:
    logging.info('will add asset files: %s' % asset_files)
    params['asset_files'] = asset_files
  estimator, _ = _create_estimator(pipeline_config, params=params)

  # construct serving input fn
  export_config = pipeline_config.export_config
  serving_input_fn = _get_input_fn(data_config, feature_configs, None,
                                   export_config, **input_fn_kwargs)
  ckpt_path = _get_ckpt_path(pipeline_config, checkpoint_path)
  estimator.export_checkpoint(
      export_path=export_path,
      serving_input_receiver_fn=serving_input_fn,
      checkpoint_path=ckpt_path,
      mode=mode)

  logging.info('model checkpoint has been exported successfully')
