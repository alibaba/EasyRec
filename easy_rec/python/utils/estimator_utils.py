# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import re
import time
from distutils.version import LooseVersion

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import meta_graph

from easy_rec.python.protos.eas_serving_pb2 import EmbeddingPartData
from easy_rec.python.utils import shape_utils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
  SessionRunHook = tf.estimator.SessionRunHook
  CheckpointSaverHook = tf.estimator.CheckpointSaverHook
else:
  SessionRunHook = tf.train.SessionRunHook
  CheckpointSaverHook = tf.train.CheckpointSaverHook


class ExitBarrierHook(SessionRunHook):
  """ExitBarrier to make sure master and workers exit at the same time.

  After training finish, master has to do evaluation and model export, so master exits a little late
  than workers.
  """

  def __init__(self, num_worker, is_chief, model_dir):
    self._num_worker = num_worker
    self._is_chief = is_chief
    self._queue = None
    self._signal_que = None
    self._que_size = None
    self._queue = None
    self._enque = None
    self._deque = None
    self._model_dir = model_dir
    self._send = None
    self._recv = None

  def begin(self):
    """Count the number of workers and masters, and setup barrier queue."""
    tf.logging.info('number workers(including master) = %d' % self._num_worker)
    with tf.device(
        tf.DeviceSpec(job='ps', task=0, device_type='CPU', device_index=0)):
      self._queue = tf.FIFOQueue(
          capacity=self._num_worker,
          dtypes=[tf.float32],
          shapes=[()],
          name='exit_counter',
          shared_name='exit_counter')
      self._signal_que = tf.FIFOQueue(
          capacity=self._num_worker,
          dtypes=[tf.string],
          shapes=[()],
          name='exit_counter_signal',
          shared_name='exit_counter_signal')
    self._enque = self._queue.enqueue(1.0)
    self._que_size = self._queue.size()
    self._deque = self._queue.dequeue()
    if self._is_chief:
      self._flag_file = os.path.join(self._model_dir,
                                     'atexit_sync_' + str(int(time.time())))
      self._send = self._signal_que.enqueue([self._flag_file])
    else:
      self._recv = self._signal_que.dequeue()
      self._flag_file = None

  def after_create_session(self, session, coord):
    """Clean up the queue after create session.

    Sometimes ps is not exit, the last run enqueued elements will remain in the queue
    """
    if self._is_chief:
      # clear the queue
      que_size = session.run(self._que_size)
      while que_size > 0:
        session.run(self._deque)
        que_size = session.run(self._que_size)
      logging.info('exit counter cleared: %d' % que_size)

  def end(self, session):
    """Ensure when all workers and master enqueue an element, then exit."""
    session.run(self._enque)
    que_size = session.run(self._que_size)
    while que_size < self._num_worker:
      que_size = session.run(self._que_size)
      time.sleep(5)
      tf.logging.info(
          'waiting for other worker to exit, finished %d, total %d' %
          (que_size, self._num_worker))
    # prepare on_exit synchronize base on self._flag_file
    if self._is_chief:
      for i in range(self._num_worker - 1):
        session.run(self._send)
    else:
      self._flag_file = session.run(self._recv)

    def _check_flag_file(is_chief, flag_file):
      logging.info('_check_flag_file: is_chief = %d flag_file=%s' %
                   (is_chief, flag_file))
      if is_chief:
        with tf.gfile.GFile(flag_file, 'w') as fout:
          fout.write('atexit time: %d' % int(time.time()))
      else:
        while not tf.gfile.Exists(flag_file):
          time.sleep(1)

    from atexit import register
    register(
        _check_flag_file, is_chief=self._is_chief, flag_file=self._flag_file)
    logging.info('ExitBarrier passed')


class ProgressHook(SessionRunHook):

  def __init__(self, num_steps, filename, is_chief):
    """Initializes a `ProgressHook`.

    Args:
      num_steps: total train steps
      filename: progress file name
      is_chief: is chief worker or not
    """
    self._num_steps = num_steps
    self._is_chief = is_chief
    if self._is_chief:
      self._progress_file = tf.gfile.GFile(filename, 'w')
      self._progress_file.write('0.00\n')
      self._progress_interval = 0.01  # 1%
      self._last_progress_cnt = 0

  def before_run(self, run_context):
    if self._is_chief:
      return tf.train.SessionRunArgs([tf.train.get_global_step()])

  def after_run(
      self,
      run_context,  # pylint: disable=unused-argument
      run_values):
    if self._is_chief:
      global_step = run_values.results[0]
      curr_progress = global_step / self._num_steps
      curr_progress_cnt = int(curr_progress / self._progress_interval)
      if curr_progress_cnt >= self._last_progress_cnt + 1:
        self._progress_file.write('%.2f\n' % curr_progress)
        self._progress_file.flush()
        self._last_progress_cnt = curr_progress_cnt
        logging.info('Training Progress: %.2f' % curr_progress)

  def end(self, session):
    if self._is_chief:
      if self._last_progress_cnt < 1 / self._progress_interval:
        self._progress_file.write('1.00\n')
      self._progress_file.close()


class EmbeddingPartSaver:
  """Large Embedding Saver.

  For large embedding serving on eas, large embeddings are partitioned and saved separately.
  """

  def __init__(self, var):
    self._var = var
    # normalize var names
    var_name = var.name.split('/')
    if var_name[-2] == 'embedding_weights' and 'part_' in var_name[-1]:
      # input_layer uid embedding_weights part_0:0
      part_name = var_name[-1].split(':')[0]
      var_name = var_name[-3]
      if var_name.endswith('_embedding'):
        var_name = var_name[:-len('_embedding')]
      var_name = var_name + '.' + part_name
    else:
      # input_layer tag embedding_weights:0
      var_name = var_name[-2]
      if var_name.endswith('_embedding'):
        var_name = var_name[:-len('_embedding')]
    logging.info('embedding variable name: %s normalize_name: %s' %
                 (var.name, var_name))
    self._var_name = var_name

  @property
  def name(self):
    return self._var_name

  def save(self, session, save_path, global_step):
    """Save embedding data as EmbeddingPartData as .pb files.

    Args:
      session: tf.Session instance
      save_path: data save path
      global_step: train step
    """
    var_data = session.run(self._var)
    embed_part_data = EmbeddingPartData()

    for x in var_data.shape:
      embed_part_data.shape.append(x)

    for r in range(var_data.shape[0]):
      for c in range(var_data.shape[1]):
        embed_part_data.data.append(var_data[r, c])

    save_path = save_path + '.pb.' + str(global_step)
    with tf.gfile.GFile(save_path, 'wb') as fout:
      fout.write(embed_part_data.SerializeToString())
    logging.info('save embedding %s to %s done' % (self._var_name, save_path))


class CheckpointSaverHook(CheckpointSaverHook):
  """Saves checkpoints every N steps or seconds."""

  def __init__(self,
               checkpoint_dir,
               save_secs=None,
               save_steps=None,
               saver=None,
               checkpoint_basename='model.ckpt',
               scaffold=None,
               listeners=None,
               write_graph=True,
               separate_save=False):
    """Initializes a `CheckpointSaverHook`.

    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      save_secs: `int`, save every N secs.
      save_steps: `int`, save every N steps.
      saver: `Saver` object, used for saving.
      checkpoint_basename: `str`, base name for the checkpoint files.
      scaffold: `Scaffold`, use to get saver object.
      listeners: List of `CheckpointSaverListener` subclass instances.
        Used for callbacks that run immediately before or after this hook saves
        the checkpoint.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of saver or scaffold should be set.
    """
    super(CheckpointSaverHook, self).__init__(
        checkpoint_dir,
        save_secs=save_secs,
        save_steps=save_steps,
        saver=saver,
        checkpoint_basename=checkpoint_basename,
        scaffold=scaffold,
        listeners=listeners)
    self._write_graph = write_graph

  def after_create_session(self, session, coord):
    global_step = session.run(self._global_step_tensor)
    if self._write_graph:
      # We do write graph and saver_def at the first call of before_run.
      # We cannot do this in begin, since we let other hooks to change graph and
      # add variables in begin. Graph is finalized after all begin calls.
      tf.train.write_graph(tf.get_default_graph().as_graph_def(add_shapes=True),
                           self._checkpoint_dir, 'graph.pbtxt')
      saver_def = self._get_saver().saver_def if self._get_saver() else None
      graph = tf.get_default_graph()
      meta_graph_def = meta_graph.create_meta_graph_def(
          graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
      self._summary_writer.add_graph(graph)
      self._summary_writer.add_meta_graph(meta_graph_def)
    # when tf version > 1.10.0, we use defaut training strategy, which saves ckpt
    # at first train step
    if LooseVersion(tf.__version__) >= LooseVersion('1.10.0'):
      # The checkpoint saved here is the state at step "global_step".
      self._save(session, global_step)
    self._timer.update_last_triggered_step(global_step)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def _save(self, session, step):
    """Saves the latest checkpoint, returns should_stop."""
    logging.info('Saving checkpoints for %d into %s.', step, self._save_path)

    for l in self._listeners:  # noqa: E741
      l.before_save(session, step)

    self._get_saver().save(
        session,
        self._save_path,
        global_step=step,
        write_meta_graph=self._write_graph)
    save_dir, save_name = os.path.split(self._save_path)
    save_dir = os.path.join(save_dir, 'embeddings')
    if not tf.gfile.Exists(save_dir):
      tf.gfile.MakeDirs(save_dir)

    self._summary_writer.add_session_log(
        tf.SessionLog(
            status=tf.SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
        step)

    should_stop = False
    for l in self._listeners:  # noqa: E741
      if l.after_save(session, step):
        logging.info(
            'A CheckpointSaverListener requested that training be stopped. '
            'listener: {}'.format(l))
        should_stop = True
    return should_stop


class NumpyCheckpointRestoreHook(SessionRunHook):
  """Restore variable from numpy checkpoint."""

  def __init__(self, ckpt_path, name2var_map):
    """Initializes a `NumpyCheckpointRestoreHook`.

    Args:
      ckpt_path: numpy checkpoint path to restore from
      name2var_map: var name in numpy ckpt to variable map
    """
    self._ckpt_path = ckpt_path
    self._name2var_map = name2var_map
    self._restore_op = None

  def begin(self):
    ckpt_data = np.load(self._ckpt_path)
    vars_not_inited = {}

    assign_ops = []
    has_shape_unmatch = False
    with tf.variable_scope('', reuse=True):
      for var_name, var in six.iteritems(self._name2var_map):
        var_shape = var.get_shape().as_list()
        if var_name in ckpt_data.keys():
          var_data = ckpt_data[var_name]
          if list(var_data.shape) == var_shape:
            assign_ops.append(var.assign(var_data))
          else:
            logging.error(
                'variable [%s] shape not match %r vs %r' %
                (var.name.split(':')[0], var_shape, list(var_data.shape)))
            has_shape_unmatch = True
        elif 'Momentum' not in var_name and 'global_step' not in var_name:
          logging.error('variable [%s] not found in ckpt' % var_name)
          vars_not_inited[var_name] = ','.join([str(s) for s in var_shape])
    self._restore_op = tf.group(assign_ops)

    with tf.gfile.GFile(self._ckpt_path[:-4] + '_not_inited.txt', 'w') as f:
      for var_name in sorted(vars_not_inited.keys()):
        f.write('%s:%s\n' % (var_name, vars_not_inited[var_name]))
    assert not has_shape_unmatch, 'exist variable shape not match, restore failed'
    assert len(vars_not_inited.keys()) == 0, \
        'exist variable shape not inited, restore failed'

  def after_create_session(self, session, coord):
    assert self._restore_op is not None
    logging.info('running numpy checkpoint restore_op')
    session.run(self._restore_op)


class IncompatibleShapeRestoreHook(SessionRunHook):
  """Restore variable with incompatible shapes."""

  def __init__(self, incompatible_shape_var_map):
    """Initializes a `IncompatibleShapeRestoreHook`.

    Args:
      incompatible_shape_var_map: a variables mapping with incompatible shapes,
        map from real variable to temp variable, real variable is the variable
        used in model, temp variable is the variable restored from checkpoint.
    """
    self._incompatible_shape_var_map = incompatible_shape_var_map
    self._restore_op = None

  def begin(self):
    assign_ops = []
    for var, var_tmp in six.iteritems(self._incompatible_shape_var_map):
      assign_ops.append(
          var.assign(
              shape_utils.pad_or_clip_nd(var_tmp,
                                         var.get_shape().as_list())))
      logging.info(
          'Assign variable[%s] from shape%s to shape%s' %
          (var.name, var_tmp.get_shape().as_list(), var.get_shape().as_list()))
    self._restore_op = tf.group(assign_ops)

  def after_create_session(self, session, coord):
    assert self._restore_op is not None
    logging.info('running incompatible shape variable restore_op')
    session.run(self._restore_op)


class MultipleCheckpointsRestoreHook(SessionRunHook):
  """Restore variable from numpy checkpoint."""
  SEP = ';'

  def __init__(self, ckpt_paths):
    """Initializes a `MultipleCheckpointsRestoreHook`.

    Args:
      ckpt_paths: multiple checkpoint path, seperated by ;
      name2var_map: var name in numpy ckpt to variable map
    """
    self._ckpt_path_list = ckpt_paths.split(self.SEP)
    self._saver_list = []

  def begin(self):
    global_variables = tf.global_variables()
    var_names = [re.sub(':[0-9]$', '', var.name) for var in global_variables]
    restore_status = {var_name: False for var_name in var_names}
    for ckpt_path in self._ckpt_path_list:
      logging.info('read variable from %s' % ckpt_path)
      ckpt_reader = tf.train.NewCheckpointReader(ckpt_path)
      ckpt_var2shape_map = ckpt_reader.get_variable_to_shape_map()
      # ckpt_var2shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
      name2var = {}
      for var in global_variables:
        var_name = re.sub(':[0-9]$', '', var.name)
        if var_name in ckpt_var2shape_map:
          if restore_status[var_name]:
            logging.warning(
                'variable %s find in more than one checkpoint, skipped %s' %
                (var_name, ckpt_path))
            continue
          name2var[var_name] = var
          restore_status[var_name] = True
      saver = tf.train.Saver(name2var)
      self._saver_list.append(saver)

    restore_check = True
    for var_name, stat in six.iteritems(restore_status):
      if not stat:
        logging.error('var %s not find in checkpoints' % var_name)
        restore_check = False

    assert restore_check, 'failed to find all variables in checkpoints provided'

  def after_create_session(self, session, coord):
    logging.info('running multiple checkpoint restore hook')
    for saver, ckpt_path in zip(self._saver_list, self._ckpt_path_list):
      logging.info('restore checkpoint from %s' % ckpt_path)
      saver.restore(session, ckpt_path)


def parse_tf_config():
  tf_config_str = os.environ.get('TF_CONFIG', '')
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(tf_config_str)
    cluster = tf_config['cluster']
    task = tf_config['task']
    task_type = task['type']
    task_index = task['index']
  else:
    cluster = {}
    task_type = 'master'
    task_index = 0
  return cluster, task_type, task_index


def get_task_index_and_num():
  cluster, task_type, task_index = parse_tf_config()
  if 'worker' not in cluster:
    return 0, 1
  if task_type == 'evaluator':
    return 0, 1

  task_num = len(cluster['worker'])
  if 'chief' in cluster or 'master' in cluster:
    task_num += 1
    if task_type not in ['chief', 'master']:
      task_index += 1
  return task_index, task_num
