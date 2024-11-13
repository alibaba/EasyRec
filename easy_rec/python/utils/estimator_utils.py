# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import re
import sys
import time

import numpy as np
import six
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.client import device_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.summary_io import SummaryWriterCache

from easy_rec.python.ops.incr_record import get_sparse_indices
from easy_rec.python.ops.incr_record import kv_resource_incr_gather
from easy_rec.python.utils import constant
from easy_rec.python.utils import embedding_utils
from easy_rec.python.utils import shape_utils

from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer  # NOQA

try:
  import horovod.tensorflow as hvd
except Exception:
  hvd = None

try:
  from sparse_operation_kit import experiment as sok
except Exception:
  sok = None

try:
  from kafka import KafkaProducer, KafkaAdminClient
  from kafka.admin import NewTopic
except ImportError as ex:
  logging.warning('kafka-python is not installed: %s' % str(ex))

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
SessionRunHook = session_run_hook.SessionRunHook
CheckpointSaverHook = basic_session_run_hooks.CheckpointSaverHook


def tensor_log_format_func(tensor_dict):
  prefix = ''
  if 'step' in tensor_dict:
    prefix = 'global step %s: ' % tensor_dict['step']
  stats = []
  for k in tensor_dict:
    if k == 'step':
      continue
    tensor_value = tensor_dict[k]
    stats.append('%s = %s' % (k, tensor_value))
  return prefix + ', '.join(stats)


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
        with gfile.GFile(flag_file, 'w') as fout:
          fout.write('atexit time: %d' % int(time.time()))
      else:
        while not gfile.Exists(flag_file):
          time.sleep(1)

    from atexit import register
    register(
        _check_flag_file, is_chief=self._is_chief, flag_file=self._flag_file)
    logging.info('ExitBarrier passed')


class EvaluateExitBarrierHook(SessionRunHook):
  """ExitBarrier to make sure master and workers exit at the same time.

  After training finish, master has to do evaluation and model export, so master exits a little late
  than workers.
  """

  def __init__(self, num_worker, is_chief, model_dir, metric_ops=None):
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
    self.metric_ops = metric_ops
    self.eval_result = None

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
      self.eval_result = session.run(self.metric_ops)
      for i in range(self._num_worker - 1):
        session.run(self._send)
    else:
      self._flag_file = session.run(self._recv)

    def _check_flag_file(is_chief, flag_file):
      logging.info('_check_flag_file: is_chief = %d flag_file=%s' %
                   (is_chief, flag_file))
      if is_chief:
        with gfile.GFile(flag_file, 'w') as fout:
          fout.write('atexit time: %d' % int(time.time()))
      else:
        while not gfile.Exists(flag_file):
          time.sleep(1)

    from atexit import register
    register(
        _check_flag_file, is_chief=self._is_chief, flag_file=self._flag_file)
    session.run(self.metric_ops)

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
      self._progress_file = gfile.GFile(filename, 'w')
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
               data_offset_var=None,
               increment_save_config=None):
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
      write_graph: whether to save graph.pbtxt.
      data_offset_var: data offset variable.
      increment_save_config: parameters for saving increment checkpoints.

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
    self._cuda_profile_start = 0
    self._cuda_profile_stop = 0
    self._steps_per_run = 1
    self._write_graph = write_graph
    self._data_offset_var = data_offset_var

    self._task_idx, self._task_num = get_task_index_and_num()

    if increment_save_config is not None:
      self._kafka_timeout_ms = os.environ.get('KAFKA_TIMEOUT', 600) * 1000
      logging.info('KAFKA_TIMEOUT: %dms' % self._kafka_timeout_ms)
      self._kafka_max_req_size = os.environ.get('KAFKA_MAX_REQ_SIZE',
                                                1024 * 1024 * 64)
      logging.info('KAFKA_MAX_REQ_SIZE: %d' % self._kafka_max_req_size)
      self._kafka_max_msg_size = os.environ.get('KAFKA_MAX_MSG_SIZE',
                                                1024 * 1024 * 1024)
      logging.info('KAFKA_MAX_MSG_SIZE: %d' % self._kafka_max_msg_size)

      self._dense_name_to_ids = embedding_utils.get_dense_name_to_ids()
      self._sparse_name_to_ids = embedding_utils.get_sparse_name_to_ids()

      with gfile.GFile(
          os.path.join(checkpoint_dir, constant.DENSE_UPDATE_VARIABLES),
          'w') as fout:
        json.dump(self._dense_name_to_ids, fout, indent=2)

      save_secs = increment_save_config.dense_save_secs
      save_steps = increment_save_config.dense_save_steps
      self._dense_timer = SecondOrStepTimer(
          every_secs=save_secs if save_secs > 0 else None,
          every_steps=save_steps if save_steps > 0 else None)
      save_secs = increment_save_config.sparse_save_secs
      save_steps = increment_save_config.sparse_save_steps
      self._sparse_timer = SecondOrStepTimer(
          every_secs=save_secs if save_secs > 0 else None,
          every_steps=save_steps if save_steps > 0 else None)

      self._dense_timer.update_last_triggered_step(0)
      self._sparse_timer.update_last_triggered_step(0)

      self._sparse_indices = []
      self._sparse_values = []
      sparse_train_vars = ops.get_collection(constant.SPARSE_UPDATE_VARIABLES)
      for sparse_var, indice_dtype in sparse_train_vars:
        with ops.control_dependencies([tf.train.get_global_step()]):
          with ops.colocate_with(sparse_var):
            sparse_indice = get_sparse_indices(
                var_name=sparse_var.op.name, ktype=indice_dtype)
          # sparse_indice = sparse_indice.global_indices
        self._sparse_indices.append(sparse_indice)
        if 'EmbeddingVariable' in str(type(sparse_var)):
          self._sparse_values.append(
              kv_resource_incr_gather(
                  sparse_var._handle, sparse_indice,
                  np.zeros(sparse_var.shape.as_list(), dtype=np.float32)))
          # sparse_var.sparse_read(sparse_indice))
        else:
          self._sparse_values.append(
              array_ops.gather(sparse_var, sparse_indice))

      self._kafka_producer = None
      self._incr_save_dir = None
      if increment_save_config.HasField('kafka'):
        self._topic = increment_save_config.kafka.topic
        logging.info('increment save topic: %s' % self._topic)

        admin_clt = KafkaAdminClient(
            bootstrap_servers=increment_save_config.kafka.server,
            request_timeout_ms=self._kafka_timeout_ms,
            api_version_auto_timeout_ms=self._kafka_timeout_ms)
        if self._topic not in admin_clt.list_topics():
          admin_clt.create_topics(
              new_topics=[
                  NewTopic(
                      name=self._topic,
                      num_partitions=1,
                      replication_factor=1,
                      topic_configs={
                          'max.message.bytes': self._kafka_max_msg_size
                      })
              ],
              validate_only=False)
        logging.info('create increment save topic: %s' % self._topic)
        admin_clt.close()

        servers = increment_save_config.kafka.server.split(',')
        self._kafka_producer = KafkaProducer(
            bootstrap_servers=servers,
            max_request_size=self._kafka_max_req_size,
            api_version_auto_timeout_ms=self._kafka_timeout_ms,
            request_timeout_ms=self._kafka_timeout_ms)
      elif increment_save_config.HasField('fs'):
        fs = increment_save_config.fs
        if fs.relative:
          self._incr_save_dir = os.path.join(checkpoint_dir, fs.incr_save_dir)
        else:
          self._incr_save_dir = fs.incr_save_dir
        if not self._incr_save_dir.endswith('/'):
          self._incr_save_dir += '/'
        if not gfile.IsDirectory(self._incr_save_dir):
          gfile.MakeDirs(self._incr_save_dir)
      elif increment_save_config.HasField('datahub'):
        raise NotImplementedError('datahub increment saving is in development.')
      else:
        raise ValueError(
            'incr_update not specified correctly, must be oneof: kafka,fs')

      self._debug_save_update = increment_save_config.debug_save_update
    else:
      self._dense_timer = None
      self._sparse_timer = None

  def after_create_session(self, session, coord):
    global_step = session.run(self._global_step_tensor)
    if self._write_graph:
      # We do write graph and saver_def at the first call of before_run.
      # We cannot do this in begin, since we let other hooks to change graph and
      # add variables at begin. Graph is finalized after all begin calls.
      tf.train.write_graph(tf.get_default_graph().as_graph_def(add_shapes=True),
                           self._checkpoint_dir, 'graph.pbtxt')
      saver_def = self._get_saver().saver_def if self._get_saver() else None
      graph = tf.get_default_graph()
      meta_graph_def = meta_graph.create_meta_graph_def(
          graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
      self._summary_writer.add_graph(graph)
      self._summary_writer.add_meta_graph(meta_graph_def)

    # save for step 0
    self._save(session, global_step)

    self._timer.update_last_triggered_step(global_step)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def _send_dense(self, global_step, session):
    dense_train_vars = ops.get_collection(constant.DENSE_UPDATE_VARIABLES)
    dense_train_vals = session.run(dense_train_vars)
    logging.info('global_step=%d, increment save dense variables' % global_step)

    # build msg header
    msg_num = len(dense_train_vals)
    msg_ids = [self._dense_name_to_ids[x.op.name] for x in dense_train_vars]
    # 0 mean dense update message
    msg_header = [0, msg_num, global_step]
    for msg_id, x in zip(msg_ids, dense_train_vals):
      msg_header.append(msg_id)
      msg_header.append(x.size)

    # build msg body
    bytes_buf = np.array(msg_header, dtype=np.int32).tobytes()
    for x in dense_train_vals:
      bytes_buf += x.tobytes()

    if self._kafka_producer is not None:
      msg_key = 'dense_update_%d' % global_step
      send_res = self._kafka_producer.send(
          self._topic, bytes_buf, key=msg_key.encode('utf-8'))
      logging.info('kafka send dense: %d exception: %s' %
                   (global_step, send_res.exception))

    if self._incr_save_dir is not None:
      save_path = os.path.join(self._incr_save_dir,
                               'dense_update_%d' % global_step)
      with gfile.GFile(save_path, 'wb') as fout:
        fout.write(bytes_buf)
      save_flag = save_path + '.done'
      with gfile.GFile(save_flag, 'w') as fout:
        fout.write('dense_update_%d' % global_step)

    if self._debug_save_update and self._incr_save_dir is None:
      base_dir, _ = os.path.split(self._save_path)
      incr_save_dir = os.path.join(base_dir, 'incr_save/')
      if not gfile.Exists(incr_save_dir):
        gfile.MakeDirs(incr_save_dir)
      save_path = os.path.join(incr_save_dir, 'dense_update_%d' % global_step)
      with gfile.GFile(save_path, 'wb') as fout:
        fout.write(bytes_buf)

    logging.info(
        'global_step=%d, increment update dense variables, msg_num=%d' %
        (global_step, msg_num))

  def _send_sparse(self, global_step, session):
    sparse_train_vars = ops.get_collection(constant.SPARSE_UPDATE_VARIABLES)
    sparse_res = session.run(self._sparse_indices + self._sparse_values)
    msg_num = int(len(sparse_res) / 2)

    sel_ids = [i for i in range(msg_num) if len(sparse_res[i]) > 0]
    sparse_key_res = [sparse_res[i] for i in sel_ids]
    sparse_val_res = [sparse_res[i + msg_num] for i in sel_ids]
    sparse_train_vars = [sparse_train_vars[i][0] for i in sel_ids]

    sel_embed_ids = [
        self._sparse_name_to_ids[x.name] for x in sparse_train_vars
    ]

    msg_num = len(sel_ids)

    if msg_num == 0:
      logging.warning('there are no sparse updates, will skip this send: %d' %
                      global_step)
      return

    # build msg header
    # 1 means sparse update messages
    msg_header = [1, msg_num, global_step]
    for tmp_id, tmp_key in zip(sel_embed_ids, sparse_key_res):
      msg_header.append(tmp_id)
      msg_header.append(len(tmp_key))
    bytes_buf = np.array(msg_header, dtype=np.int32).tobytes()

    # build msg body
    for tmp_id, tmp_key, tmp_val, tmp_var in zip(sel_embed_ids, sparse_key_res,
                                                 sparse_val_res,
                                                 sparse_train_vars):
      # for non kv embedding variables, add partition offset to tmp_key
      if 'EmbeddingVariable' not in str(type(tmp_var)):
        if tmp_var._save_slice_info is not None:
          tmp_key += tmp_var._save_slice_info.var_offset[0]
      bytes_buf += tmp_key.tobytes()
      bytes_buf += tmp_val.tobytes()
    if self._kafka_producer is not None:
      msg_key = 'sparse_update_%d' % global_step
      send_res = self._kafka_producer.send(
          self._topic, bytes_buf, key=msg_key.encode('utf-8'))
      logging.info('kafka send sparse: %d %s' %
                   (global_step, send_res.exception))

    if self._incr_save_dir is not None:
      save_path = os.path.join(self._incr_save_dir,
                               'sparse_update_%d' % global_step)
      with gfile.GFile(save_path, 'wb') as fout:
        fout.write(bytes_buf)
      save_flag = save_path + '.done'
      with gfile.GFile(save_flag, 'w') as fout:
        fout.write('sparse_update_%d' % global_step)

    if self._debug_save_update and self._incr_save_dir is None:
      base_dir, _ = os.path.split(self._save_path)
      incr_save_dir = os.path.join(base_dir, 'incr_save/')
      if not gfile.Exists(incr_save_dir):
        gfile.MakeDirs(incr_save_dir)
      save_path = os.path.join(incr_save_dir, 'sparse_update_%d' % global_step)
      with gfile.GFile(save_path, 'wb') as fout:
        fout.write(bytes_buf)

    logging.info(
        'global_step=%d, increment update sparse variables, msg_num=%d, msg_size=%d'
        % (global_step, msg_num, len(bytes_buf)))

  def after_run(self, run_context, run_values):
    super(CheckpointSaverHook, self).after_run(run_context, run_values)
    stale_global_step = run_values.results
    global_step = -1
    if self._dense_timer is not None and self._dense_timer.should_trigger_for_step(
        stale_global_step + self._steps_per_run):
      global_step = run_context.session.run(self._global_step_tensor)
      self._dense_timer.update_last_triggered_step(global_step)
      self._send_dense(global_step, run_context.session)

    if self._sparse_timer is not None and self._sparse_timer.should_trigger_for_step(
        stale_global_step + self._steps_per_run):
      if global_step < 0:
        global_step = run_context.session.run(self._global_step_tensor)

      self._sparse_timer.update_last_triggered_step(global_step)
      self._send_sparse(global_step, run_context.session)

  def _save(self, session, step):
    """Saves the latest checkpoint, returns should_stop."""
    logging.info('Saving checkpoints for %d into %s.', step, self._save_path)

    for l in self._listeners:  # noqa: E741
      l.before_save(session, step)

    if self._data_offset_var is not None:
      save_data_offset = session.run(self._data_offset_var)
      data_offset_json = {}
      for x in save_data_offset:
        if x:
          data_offset_json.update(json.loads(x))
      save_offset_path = os.path.join(self._checkpoint_dir,
                                      'model.ckpt-%d.offset' % step)
      with gfile.GFile(save_offset_path, 'w') as fout:
        json.dump(data_offset_json, fout)

    self._get_saver().save(
        session,
        self._save_path,
        global_step=step,
        write_meta_graph=self._write_graph)

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

  def end(self, session):
    global_step = session.run(self._global_step_tensor)
    super(CheckpointSaverHook, self).end(session)
    if self._dense_timer is not None and \
        global_step != self._dense_timer.last_triggered_step():
      self._dense_timer.update_last_triggered_step(global_step)
      self._send_dense(global_step, session)
    if self._sparse_timer is not None and \
        global_step != self._sparse_timer.last_triggered_step():
      self._sparse_timer.update_last_triggered_step(global_step)
      self._send_sparse(global_step, session)


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

    with gfile.GFile(self._ckpt_path[:-4] + '_not_inited.txt', 'w') as f:
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


class OnlineEvaluationHook(SessionRunHook):

  def __init__(self, metric_dict, output_dir):
    self._metric_dict = metric_dict
    self._output_dir = output_dir
    self._summary_writer = SummaryWriterCache.get(self._output_dir)

  def end(self, session):
    metric_tensor_dict = {k: v[0] for k, v in self._metric_dict.items()}
    metric_value_dict = session.run(metric_tensor_dict)
    tf.logging.info('Eval metric: %s' % metric_value_dict)

    global_step_tensor = tf.train.get_or_create_global_step()
    global_step = session.run(global_step_tensor)

    summary = Summary()
    for k, v in metric_value_dict.items():
      summary.value.add(tag=k, simple_value=v)
    self._summary_writer.add_summary(summary, global_step=global_step)
    self._summary_writer.flush()

    eval_result_file = os.path.join(self._output_dir,
                                    'online_eval_result.txt-%s' % global_step)
    logging.info('Saving online eval result to file %s' % eval_result_file)
    with gfile.GFile(eval_result_file, 'w') as ofile:
      result_to_write = {}
      for key in sorted(metric_value_dict):
        # convert numpy float to python float
        result_to_write[key] = metric_value_dict[key].item()
      ofile.write(json.dumps(result_to_write, indent=2))


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
  if hvd is not None and 'HOROVOD_RANK' in os.environ:
    return hvd.rank(), hvd.size()
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


def get_ckpt_version(ckpt_path):
  """Get checkpoint version from ckpt_path.

  Args:
    ckpt_path: such as xx/model.ckpt-2000 or xx/model.ckpt-2000.meta

  Return:
    ckpt_version: such as 2000
  """
  _, ckpt_name = os.path.split(ckpt_path)
  ckpt_name, ext = os.path.splitext(ckpt_name)
  if ext.startswith('.ckpt-'):
    ckpt_name = ext
  toks = ckpt_name.split('-')
  return int(toks[-1])


def get_latest_checkpoint_from_checkpoint_path(checkpoint_path,
                                               ignore_ckpt_error):
  ckpt_path = None
  if checkpoint_path.endswith('/') or gfile.IsDirectory(checkpoint_path + '/'):
    checkpoint_dir = checkpoint_path
    if not checkpoint_dir.endswith('/'):
      checkpoint_dir = checkpoint_dir + '/'
    if gfile.Exists(checkpoint_dir):
      ckpt_path = latest_checkpoint(checkpoint_dir)
      if ckpt_path:
        logging.info(
            'fine_tune_checkpoint is directory, will use the latest checkpoint: %s'
            % ckpt_path)
      else:
        assert ignore_ckpt_error, 'fine_tune_checkpoint(%s) is not exists.' % checkpoint_path
    else:
      assert ignore_ckpt_error, 'fine_tune_checkpoint(%s) is not exists.' % checkpoint_path
  elif gfile.Exists(checkpoint_path + '.index'):
    ckpt_path = checkpoint_path
    logging.info('update fine_tune_checkpoint to %s' % checkpoint_path)
  else:
    assert ignore_ckpt_error, 'fine_tune_checkpoint(%s) is not exists.' % checkpoint_path
  return ckpt_path


def latest_checkpoint(model_dir):
  """Find lastest checkpoint under a directory.

  Args:
    model_dir: model directory

  Return:
    model_path: xx/model.ckpt-2000
  """
  try:
    ckpt_metas = gfile.Glob(os.path.join(model_dir, 'model.ckpt-*.index'))

    if len(ckpt_metas) == 0:
      return None

    if len(ckpt_metas) > 1:
      ckpt_metas.sort(key=lambda x: get_ckpt_version(x))
    ckpt_path = os.path.splitext(ckpt_metas[-1])[0]
    return ckpt_path
  except errors_impl.NotFoundError:
    return None


def get_trained_steps(model_dir):
  ckpt_path = latest_checkpoint(model_dir)
  if ckpt_path is not None:
    return int(ckpt_path.split('-')[-1])
  else:
    return 0


def master_to_chief():
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    # change chief to master
    if 'master' in tf_config['cluster']:
      tf_config['cluster']['chief'] = tf_config['cluster']['master']
      del tf_config['cluster']['chief']
      if tf_config['task']['type'] == 'master':
        tf_config['task']['type'] = 'chief'
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    return tf_config
  else:
    return None


def chief_to_master():
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    # change chief to master
    if 'chief' in tf_config['cluster']:
      tf_config['cluster']['master'] = tf_config['cluster']['chief']
      del tf_config['cluster']['chief']
      if tf_config['task']['type'] == 'chief':
        tf_config['task']['type'] = 'master'
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    return tf_config
  else:
    return None


def is_ps():
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    if 'task' in tf_config:
      return tf_config['task']['type'] == 'ps'
  return False


def is_chief():
  if has_hvd():
    return hvd.rank() == 0

  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    if 'task' in tf_config:
      return tf_config['task']['type'] in ['chief', 'master']
  return True


def is_master():
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    if 'task' in tf_config:
      return tf_config['task']['type'] == 'master'
  return True


def is_evaluator():
  if 'TF_CONFIG' in os.environ:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    if 'task' in tf_config:
      return tf_config['task']['type'] == 'evaluator'
  return False


def has_hvd():
  return hvd is not None and 'HOROVOD_RANK' in os.environ


def has_sok():
  return sok is not None and 'ENABLE_SOK' in os.environ


def init_hvd():
  if hvd is None:
    logging.error(
        'horovod is not installed: HOROVOD_WITH_TENSORFLOW=1 pip install horovod'
    )
    sys.exit(1)

  hvd.init()
  os.environ['HOROVOD_RANK'] = str(hvd.rank())


def init_sok():
  try:
    sok.init()
    os.environ['ENABLE_SOK'] = '1'
    return True
  except Exception:
    logging.warning('sok is not installed')
    return False


def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']
