# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import multiprocessing
import queue
# import threading
import time

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


class ParquetInputV2(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None,
               **kwargs):
    super(ParquetInputV2,
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
    self._data_que = queues.Queue(
        name='data_que', ctx=mp_ctxt, maxsize=self._data_config.prefetch_size)

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

    self._proc_start = False
    self._proc_start_que = queues.Queue(name='proc_start_que', ctx=mp_ctxt)
    self._proc_stop = False
    self._proc_stop_que = queues.Queue(name='proc_stop_que', ctx=mp_ctxt)

    self._reserve_fields = None
    self._reserve_types = None
    if 'reserve_fields' in kwargs and 'reserve_types' in kwargs:
      self._reserve_fields = kwargs['reserve_fields']
      self._reserve_types = kwargs['reserve_types']

    self._proc_arr = None

  def _sample_generator(self):
    if not self._proc_start:
      self._proc_start = True
      for proc in (self._proc_arr):
        self._proc_start_que.put(True)
        logging.info('task[%s] data_proc=%s is_alive=%s' %
                     (self._task_index, proc, proc.is_alive()))

    done_proc_cnt = 0
    fetch_timeout_cnt = 0

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
    while True:
      try:
        sample = self._data_que.get(timeout=1)
        if sample is None:
          done_proc_cnt += 1
        else:
          fetch_good_cnt += 1
          yield sample
        if fetch_good_cnt % 200 == 0:
          logging.info(
              'task[%d] fetch_batch_cnt=%d, fetch_timeout_cnt=%d, qsize=%d' %
              (self._task_index, fetch_good_cnt, fetch_timeout_cnt,
               self._data_que.qsize()))
      except queue.Empty:
        fetch_timeout_cnt += 1
        if done_proc_cnt >= len(self._proc_arr):
          logging.info('all sample finished, fetch_timeout_cnt=%d' %
                       fetch_timeout_cnt)
          break
      except Exception as ex:
        logging.warning('task[%d] get from data_que exception: %s' %
                        (self._task_index, str(ex)))
        break
    logging.info('task[%d] sample_generator: total_batches=%d' %
                 (self._task_index, fetch_good_cnt))

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

      def _any_alive():
        for proc in self._proc_arr:
          if proc.is_alive():
            return True
        return False

      # to ensure the sender part of the python Queue could exit
      while _any_alive():
        try:
          self._data_que.get(timeout=1)
        except Exception:
          pass
      time.sleep(1)
      self._data_que.close()
      logging.info('data que closed')
      # import time
      # time.sleep(10)
      for proc in self._proc_arr:
        # proc.terminate()
        proc.join()
      logging.info('join proc done')
      self._proc_start = False

  def _to_fea_dict(self, input_dict):
    fea_dict = {}

    # if self._has_ev:
    #   tmp_vals, tmp_lens = input_dict['sparse_fea'][1], input_dict[
    #       'sparse_fea'][0]

    #   fea_dict['sparse_fea'] = (tmp_vals, tmp_lens)
    # else:
    #   tmp_vals, tmp_lens = input_dict['sparse_fea'][1], input_dict[
    #       'sparse_fea'][0]
    #   fea_dict['sparse_fea'] = (tmp_vals % self._feature_configs[0].num_buckets,
    #                             tmp_lens)

    for fea in self._effective_fields:
      tmp_lens, tmp_vals = input_dict[fea]
      if self._has_ev:
        fea_dict[fea] = tf.RaggedTensor.from_row_lengths(
            values=tmp_vals, row_lengths=tmp_lens)
      else:
        fea_dict[fea] = tf.RaggedTensor.from_row_lengths(
            values=(tmp_vals % self._feature_configs[0].num_buckets),
            row_lengths=tmp_lens)

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

    if mode == tf.estimator.ModeKeys.TRAIN:
      self._proc_arr = load_parquet.start_data_proc(
          self._task_index,
          self._task_num,
          self._num_proc,
          self._file_que,
          self._data_que,
          self._proc_start_que,
          self._proc_stop_que,
          self._batch_size,
          self._label_fields,
          self._effective_fields,
          self._reserve_fields,
          self._data_config.drop_remainder,
          need_pack=False)
    else:
      lbl_fields = self._label_fields
      if mode == tf.estimator.ModeKeys.PREDICT:
        lbl_fields = None
      self._proc_arr = load_parquet.start_data_proc(
          self._task_index,
          self._task_num,
          self._num_proc,
          self._file_que,
          self._data_que,
          self._proc_start_que,
          self._proc_stop_que,
          self._batch_size,
          lbl_fields,
          self._effective_fields,
          self._reserve_fields,
          False,
          need_pack=False)

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

    for k in self._effective_fields:
      out_types[k] = (tf.int32, tf.int64)
      out_shapes[k] = (tf.TensorShape([None]), tf.TensorShape([None]))
    # out_types['sparse_fea'] = (tf.int32, tf.int64)
    # out_shapes['sparse_fea'] = (tf.TensorShape([None]), tf.TensorShape([None]))

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
