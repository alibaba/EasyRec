# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import logging
import multiprocessing
# import queue
import threading
import time
from multiprocessing import connection
from multiprocessing import context

# import numpy as np
# import pandas as pd
import tensorflow as tf
# from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import logging_ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.platform import gfile

from easy_rec.python.compat import queues
from easy_rec.python.input import load_parquet
from easy_rec.python.input.input import Input


class ParquetInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None,
               **kwargs):
    super(ParquetInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config, **kwargs)
    if input_path is None:
      return

    self._input_files = []
    for sub_path in input_path.strip().split(','):
      self._input_files.extend(gfile.Glob(sub_path))
    logging.info('parquet input_path=%s file_num=%d' %
                 (input_path, len(self._input_files)))
    mp_ctxt = multiprocessing.get_context('spawn')

    file_num = len(self._input_files)
    logging.info('[task_index=%d] total_file_num=%d task_num=%d' %
                 (task_index, file_num, task_num))

    self._my_files = []
    for file_id in range(file_num):
      if (file_id % task_num) == task_index:
        self._my_files.append(self._input_files[file_id])
    # self._my_files = self._input_files

    logging.info('[task_index=%d] task_file_num=%d' %
                 (task_index, len(self._my_files)))
    self._file_que = queues.Queue(name='file_que', ctx=mp_ctxt)

    self._num_proc = 8
    if file_num < self._num_proc:
      self._num_proc = file_num

    self._readers = []
    self._writers = []
    for proc_id in range(self._num_proc):
      reader, writer = connection.Pipe(duplex=False)
      self._readers.append(reader)
      self._writers.append(writer)

    self._proc_start = False
    self._proc_start_sem = mp_ctxt.Semaphore(0)
    self._proc_stop = False
    self._proc_stop_que = queues.Queue(name='proc_stop_que', ctx=mp_ctxt)

    self._reserve_fields = None
    self._reserve_types = None
    if 'reserve_fields' in kwargs and 'reserve_types' in kwargs:
      self._reserve_fields = kwargs['reserve_fields']
      self._reserve_types = kwargs['reserve_types']

    self._data_que = collections.deque()
    self._data_que_max_len = 32
    self._proc_arr = None
    self._recv_threads = None
    self._recv_stop = False

  def _sample_generator(self):
    if not self._proc_start:
      self._proc_start = True
      for proc in self._proc_arr:
        self._proc_start_sem.release()
        logging.info('task[%s] data_proc=%s is_alive=%s' %
                     (self._task_index, proc, proc.is_alive()))
      for recv_th in self._recv_threads:
        recv_th.start()

    done_proc_cnt = 0

    # # for mock purpose
    # all_samples = []
    # while len(all_samples) < 64:
    #   try:
    #     sample = self._data_que.get(block=False)
    #     all_samples.append(sample)
    #   except queue.Empty:
    #     continue
    # sid = 0
    # while True:
    #   yield all_samples[sid]
    #   sid += 1
    #   if sid >= len(all_samples):
    #     sid = 0

    fetch_good_cnt = 0
    fetch_fail_cnt = 0
    start_ts = time.time()
    while done_proc_cnt < len(self._proc_arr):
      try:
        if len(self._data_que) > 0:
          sample = self._data_que.popleft()
          if sample is None:
            done_proc_cnt += 1
            continue
          fetch_good_cnt += 1
          yield sample
          if fetch_good_cnt % 100 == 0 and fetch_good_cnt > 0:
            logging.info(
                'task[%d] fetch_good_cnt=%d fetch_fail_cnt=%d all_fetch_ts=%.3f que_len=%d'
                % (self._task_index, fetch_good_cnt, fetch_fail_cnt,
                   time.time() - start_ts, len(self._data_que)))
        else:
          fetch_fail_cnt += 1
      except Exception as ex:
        logging.warning(
            'task[%d] fetch_data exception: %s fetch_fail_cnt=%d que_len=%d' %
            (self._task_index, str(ex), fetch_fail_cnt, len(self._data_que)))
        break
    self._recv_stop = True

  def stop(self):
    if self._proc_arr is None or len(self._proc_arr) == 0:
      return
    logging.info('task[%d] will stop dataset procs, proc_num=%d' %
                 (self._task_index, len(self._proc_arr)))
    self._file_que.close()
    if self._proc_start:
      logging.info('try close data que')
      for _ in range(len(self._proc_arr)):
        self._proc_stop_que.put(1)
      self._proc_stop_que.close()

      self._recv_stop = True
      for recv_th in self._recv_threads:
        recv_th.join()

      def _any_alive():
        for proc in self._proc_arr:
          if proc.is_alive():
            return True
        return False

      # to ensure the sender part of the python Queue could exit
      while _any_alive():
        try:
          if self._reader.poll(1):
            self._reader.recv_bytes()
        except Exception:
          pass
      time.sleep(1)

      for reader in self._readers:
        reader.close()

      for proc in self._proc_arr:
        proc.join()
      logging.info('join proc done')
      self._proc_start = False

  def _to_fea_dict(self, input_dict):
    fea_dict = {}

    if self._has_ev:
      tmp_vals, tmp_lens = input_dict['sparse_fea'][1], input_dict[
          'sparse_fea'][0]

      fea_dict['sparse_fea'] = (tmp_vals, tmp_lens)
    else:
      tmp_vals, tmp_lens = input_dict['sparse_fea'][1], input_dict[
          'sparse_fea'][0]
      fea_dict['sparse_fea'] = (tmp_vals % self._feature_configs[0].num_buckets,
                                tmp_lens)

    output_dict = {'feature': fea_dict}

    lbl_dict = {}
    for lbl_name in self._label_fields:
      if lbl_name in input_dict:
        lbl_dict[lbl_name] = input_dict[lbl_name]

    if len(lbl_dict) > 0:
      output_dict['label'] = lbl_dict

    if self._reserve_fields is not None:
      output_dict['reserve'] = input_dict['reserve']

    return output_dict

  def _build(self, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN and self._data_config.num_epochs > 1:
      logging.info('will repeat train data for %d epochs' %
                   self._data_config.num_epochs)
      my_files = self._my_files * self._data_config.num_epochs
    else:
      my_files = self._my_files

    if mode != tf.estimator.ModeKeys.PREDICT:
      self._proc_arr = load_parquet.start_data_proc(
          self._task_index, self._task_num, self._num_proc, self._file_que,
          self._writers, self._proc_start_sem, self._proc_stop_que,
          self._batch_size, self._label_fields, self._effective_fields,
          self._reserve_fields, self._data_config.drop_remainder)
    else:
      self._proc_arr = load_parquet.start_data_proc(
          self._task_index, self._task_num, self._num_proc, self._file_que,
          self._writers, self._proc_start_sem, self._proc_stop_que,
          self._batch_size, None, self._effective_fields, self._reserve_fields,
          False)

    # create receive threads
    def _recv_func(reader, recv_id):
      start_ts = time.time()
      recv_ts = 0
      decode_ts = 0
      wait_ts = 0
      append_ts = 0
      recv_cnt = 0
      while not self._recv_stop:
        try:
          if reader.poll(0.25):
            ts0 = time.time()
            obj = reader.recv_bytes()
            ts1 = time.time()
            sample = context.reduction.ForkingPickler.loads(obj)
            ts2 = time.time()
            while len(self._data_que) >= self._data_que_max_len:
              time.sleep(0.1)
            ts3 = time.time()
            # self._data_que_lock.acquire()
            self._data_que.append(sample)
            ts4 = time.time()
            recv_ts += (ts1 - ts0)
            decode_ts += (ts2 - ts1)
            wait_ts += (ts3 - ts2)
            append_ts += (ts4 - ts3)
            recv_cnt += 1
            if recv_cnt % 100 == 0:
              logging.info((
                  'recv_time_stat[%d]: recv_ts=%.3f decode_ts=%.3f ' +
                  'wait_ts=%.3f append_ts=%.3f total=%.3f recv_cnt=%d len(data_que)=%d'
              ) % (recv_id, recv_ts, decode_ts, wait_ts, append_ts,
                   time.time() - start_ts, recv_cnt, len(self._data_que)))
            if sample is None:
              break
          else:
            continue
        except Exception as ex:
          logging.warning('recv_data exception: %s' % str(ex))
      logging.info(
          ('recv_finish, recv_time_stat[%d]: recv_ts=%.3f decode_ts=%.3f ' +
           'wait_ts=%.3f append_ts=%.3f total=%.3f recv_cnt=%d len(data_que)=%d'
           ) % (recv_id, recv_ts, decode_ts, wait_ts, append_ts,
                time.time() - start_ts, recv_cnt, len(self._data_que)))

    self._recv_threads = []
    for recv_id, reader in enumerate(self._readers):
      recv_th = threading.Thread(target=_recv_func, args=(reader, recv_id))
      self._recv_threads.append(recv_th)

    for input_file in my_files:
      self._file_que.put(input_file)

    # add end signal
    for proc in self._proc_arr:
      self._file_que.put(None)
    logging.info('add input_files to file_que, qsize=%d' %
                 self._file_que.qsize())

    out_types = {}
    out_shapes = {}

    if mode != tf.estimator.ModeKeys.PREDICT:
      for k in self._label_fields:
        out_types[k] = tf.int32
        out_shapes[k] = tf.TensorShape([None])

    if self._reserve_fields is not None:
      out_types['reserve'] = {}
      out_shapes['reserve'] = {}
      for k, t in zip(self._reserve_fields, self._reserve_types):
        out_types['reserve'][k] = t
        out_shapes['reserve'][k] = tf.TensorShape([None])

    # for k in self._effective_fields:
    #   out_types[k] = (tf.int64, tf.int32)
    #   out_shapes[k] = (tf.TensorShape([None]), tf.TensorShape([None]))
    out_types['sparse_fea'] = (tf.int32, tf.int64)
    # out_types['sparse_fea'] = (tf.int64, tf.int32)
    out_shapes['sparse_fea'] = (tf.TensorShape([None]), tf.TensorShape([None]))

    dataset = tf.data.Dataset.from_generator(
        self._sample_generator,
        output_types=out_types,
        output_shapes=out_shapes)
    num_parallel_calls = self._data_config.num_parallel_calls
    dataset = dataset.map(
        self._to_fea_dict, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    # dataset = dataset.map(
    #     map_func=self._preprocess, num_parallel_calls=num_parallel_calls)
    # dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
      # dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
      # dataset = dataset.prefetch(32)
    else:
      if self._reserve_fields is not None:
        # for predictor with saved model
        def _get_with_reserve(fea_dict):
          print('fea_dict=%s' % fea_dict)
          out_dict = {
              'feature': {
                  'ragged_ids': fea_dict['feature']['sparse_fea'][0],
                  'ragged_lens': fea_dict['feature']['sparse_fea'][1]
              }
          }
          out_dict['reserve'] = fea_dict['reserve']
          return out_dict

        dataset = dataset.map(_get_with_reserve)
      else:
        dataset = dataset.map(lambda x: self._get_features(x))
      dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    return dataset

  def create_input(self, export_config=None):

    def _input_fn(mode=None, params=None, config=None):
      """Build input_fn for estimator.

      Args:
        mode: tf.estimator.ModeKeys.(TRAIN, EVAL, PREDICT)
        params: `dict` of hyper parameters, from Estimator
        config: tf.estimator.RunConfig instance

      Return:
        if mode is not None, return:
            features: inputs to the model.
            labels: groundtruth
        else, return:
            tf.estimator.export.ServingInputReceiver instance
      """
      if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
                  tf.estimator.ModeKeys.PREDICT):
        # build dataset from self._config.input_path
        self._mode = mode
        dataset = self._build(mode, params)
        return dataset
      elif mode is None:  # serving_input_receiver_fn for export SavedModel
        ragged_ids = array_ops.placeholder(tf.int64, [None], name='ragged_ids')
        ragged_lens = array_ops.placeholder(
            tf.int32, [None], name='ragged_lens')
        inputs = {'ragged_ids': ragged_ids, 'ragged_lens': ragged_lens}
        if self._has_ev:
          features = {'ragged_ids': ragged_ids, 'ragged_lens': ragged_lens}
        else:
          features = {
              'ragged_ids': ragged_ids % self._feature_configs[0].num_buckets,
              'ragged_lens': ragged_lens
          }
        return tf.estimator.export.ServingInputReceiver(features, inputs)

    _input_fn.input_creator = self
    return _input_fn
