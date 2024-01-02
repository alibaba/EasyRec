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
    self._need_pack = True
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

    # indicator whether is called from Predictor, do not go pass
    if 'is_predictor' in kwargs:
      self._is_predictor = kwargs['is_predictor']
    else:
      self._is_predictor = False

    self._proc_arr = None

    self._sparse_fea_names = []
    self._dense_fea_names = []
    self._dense_fea_cfgs = []
    self._total_dense_fea_dim = 0
    for fc in self._feature_configs:
      feature_type = fc.feature_type
      if feature_type in [fc.IdFeature, fc.TagFeature]:
        input_name0 = fc.input_names[0]
        self._sparse_fea_names.append(input_name0)
      elif feature_type in [fc.RawFeature]:
        input_name0 = fc.input_names[0]
        self._dense_fea_names.append(input_name0)
        self._dense_fea_cfgs.append(fc)
        self._total_dense_fea_dim += fc.raw_input_dim
      else:
        assert False, 'feature_type[%s] not supported' % str(feature_type)

  def _rebuild_que(self):
    mp_ctxt = multiprocessing.get_context('spawn')
    self._data_que = queues.Queue(
        name='data_que', ctx=mp_ctxt, maxsize=self._data_config.prefetch_size)
    self._file_que = queues.Queue(name='file_que', ctx=mp_ctxt)
    self._proc_start_que = queues.Queue(name='proc_start_que', ctx=mp_ctxt)
    self._proc_stop_que = queues.Queue(name='proc_stop_que', ctx=mp_ctxt)

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

      # rebuild for next run, which is necessary for evaluation
      self._rebuild_que()
      self._proc_arr = None
      self._proc_start = False
      self._proc_stop = False

  def _to_fea_dict(self, input_dict):
    fea_dict = {}

    if len(self._sparse_fea_names) > 0:
      if self._has_ev:
        tmp_vals, tmp_lens = input_dict['sparse_fea'][1], input_dict[
            'sparse_fea'][0]

        fea_dict['sparse_fea'] = (tmp_vals, tmp_lens)
      else:
        tmp_vals, tmp_lens = input_dict['sparse_fea'][1], input_dict[
            'sparse_fea'][0]
        num_buckets = -1
        for fc in self._feature_configs:
          if fc.num_buckets > 0:
            if num_buckets < 0:
              num_buckets = fc.num_buckets
            else:
              assert num_buckets == fc.num_buckets, 'all features must share the same buckets, but are %d and %s' % (
                  num_buckets, str(fc))
        fea_dict['sparse_fea'] = (tmp_vals % num_buckets, tmp_lens)

    if len(self._dense_fea_names) > 0:
      fea_dict['dense_fea'] = input_dict['dense_fea']

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

  def add_fea_type_and_shape(self, out_types, out_shapes):
    # all features are packed into one tuple sparse_fea
    #   first field: field lengths
    #   second field: field values
    if len(self._sparse_fea_names) > 0:
      out_types['sparse_fea'] = (tf.int32, tf.int64)
      out_shapes['sparse_fea'] = (tf.TensorShape([None]), tf.TensorShape([None
                                                                          ]))
    if len(self._dense_fea_names) > 0:
      out_types['dense_fea'] = tf.float32
      out_shapes['dense_fea'] = tf.TensorShape(
          [None, self._total_dense_fea_dim])

  def _build(self, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN and self._data_config.num_epochs > 1:
      logging.info('will repeat train data for %d epochs' %
                   self._data_config.num_epochs)
      my_files = self._my_files * self._data_config.num_epochs
    else:
      my_files = self._my_files

    if mode == tf.estimator.ModeKeys.TRAIN:
      drop_remainder = self._data_config.drop_remainder
      lbl_fields = self._label_fields
    else:
      lbl_fields = self._label_fields
      if mode == tf.estimator.ModeKeys.PREDICT:
        lbl_fields = None
      drop_remainder = False
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
        # self._effective_fields,
        self._sparse_fea_names,
        self._dense_fea_names,
        self._dense_fea_cfgs,
        self._reserve_fields,
        drop_remainder,
        need_pack=self._need_pack)

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

    self.add_fea_type_and_shape(out_types, out_shapes)

    dataset = tf.data.Dataset.from_generator(
        self._sample_generator,
        output_types=out_types,
        output_shapes=out_shapes)
    num_parallel_calls = self._data_config.num_parallel_calls
    dataset = dataset.map(
        self._to_fea_dict, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    # Note: Input._preprocess is currently not supported as all features
    #      are concatenated together
    # dataset = dataset.map(
    #     map_func=self._preprocess, num_parallel_calls=num_parallel_calls)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
      # initial test show that prefetch to gpu has no performance gain
      # dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    else:
      if self._is_predictor:
        dataset = dataset.map(self._get_for_predictor)
      else:
        dataset = dataset.map(lambda x: self._get_features(x))
      dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    return dataset

  def _get_for_predictor(self, fea_dict):
    out_dict = {
        'feature': {
            'ragged_ids': fea_dict['feature']['sparse_fea'][0],
            'ragged_lens': fea_dict['feature']['sparse_fea'][1]
        }
    }
    if self._is_predictor and self._reserve_fields is not None:
      out_dict['reserve'] = fea_dict['reserve']
    return out_dict

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
        inputs, features = {}, {}
        if len(self._sparse_fea_names) > 0:
          ragged_ids = array_ops.placeholder(
              tf.int64, [None], name='ragged_ids')
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
        if len(self._dense_fea_names) > 0:
          inputs['dense_fea'] = array_ops.placeholder(
              tf.float32, [None, self._total_dense_fea_dim], name='dense_fea')
          features['dense_fea'] = inputs['dense_fea']
        return tf.estimator.export.ServingInputReceiver(features, inputs)

    _input_fn.input_creator = self
    return _input_fn
