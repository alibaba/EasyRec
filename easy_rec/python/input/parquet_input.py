# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import multiprocessing
import queue
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.input.input import Input

from easy_rec.python.input import load_parquet
from easy_rec.python.compat import queues 

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

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
    self._data_que = queues.Queue(name='data_que', ctx=mp_ctxt,
        maxsize=self._data_config.prefetch_size)
    # self._data_que._name = 'data_que'

    file_num = len(self._input_files)
    logging.info('[task_index=%d] total_file_num=%d task_num=%d' % (task_index,
        file_num, task_num))
    avg_file_num = int(file_num / task_num)
    res_file_num = file_num % task_num
    file_sid = task_index * avg_file_num + min(res_file_num, task_index)
    file_eid = (task_index + 1) * avg_file_num + min(res_file_num, task_index+1)
    self._my_files = self._input_files[file_sid:file_eid] 

    logging.info('[task_index=%d] task_file_num=%d' % (task_index, len(self._my_files)))
    self._file_que = queues.Queue(name='file_que', ctx=mp_ctxt)

    num_proc = 8
    if file_num < num_proc:
      num_proc = file_num

    self._proc_start = False
    self._proc_start_que = queues.Queue(name='proc_start_que', ctx=mp_ctxt)
    self._proc_stop = False
    self._proc_stop_que = queues.Queue(name='proc_stop_que', ctx=mp_ctxt)
    self._proc_arr = load_parquet.start_data_proc(task_index, num_proc, self._file_que,
         self._data_que, self._proc_start_que, self._proc_stop_que, self._batch_size,
         self._label_fields, self._effective_fields, self._data_config.drop_remainder)

  def _sample_generator(self):
    if not self._proc_start:
      self._proc_start = True 
      for proc in (self._proc_arr):
        self._proc_start_que.put(True)
        logging.info('task[%s] data_proc=%s is_alive=%s' % (self._task_index,
            proc, proc.is_alive()))

    done_proc_cnt = 0
    fetch_timeout_cnt = 0

    # for mock purpose
    ## all_samples = []
    ## while len(all_samples) < 64:
    ##   try:
    ##     sample = self._data_que.get(block=False)
    ##     all_samples.append(sample)
    ##   except queue.Empty:
    ##     continue
    ## sid = 0
    ## while True:
    ##   yield all_samples[sid]
    ##   sid += 1
    ##   if sid >= len(all_samples):
    ##     sid = 0

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
          logging.info('task[%d] fetch_good_cnt=%d, fetch_timeout_cnt=%d, qsize=%d' % (
              self._task_index, fetch_good_cnt, fetch_timeout_cnt, self._data_que.qsize()))
      except queue.Empty:
        fetch_timeout_cnt += 1
        if done_proc_cnt >= len(self._proc_arr):
          logging.info('all sample finished, fetch_timeout_cnt=%d' %
                       fetch_timeout_cnt)
          break
      except Exception as ex:
        logging.warning('task[%d] get from data_que exception: %s' % (self._task_index, str(ex)))
    for proc in self._proc_arr:
      proc.join()

  def stop(self):
    logging.info("task[%d] will stop dataset procs, proc_num=%d" % (self._task_index,
        len(self._proc_arr)))
    self._file_que.close()
    if self._proc_start:
      logging.info("try close data que")
      for _ in range(len(self._proc_arr)):
        self._proc_stop_que.put(1)
      self._proc_stop_que.close()

      # to ensure the sender part of the stupid python Queue could exit properly
      for _ in range(5):
        while not self._data_que.empty():
          try: 
            self._data_que.get()
          except:
            pass
        time.sleep(1)
      self._data_que.close()
      logging.info("data que closed")
      # import time
      # time.sleep(10)
      for proc in self._proc_arr:
        # proc.terminate()
        proc.join()
      logging.info("join proc done")
      self._proc_start = False
    
  def _to_fea_dict(self, input_dict):
    fea_dict = {}
    # for fea_name in self._effective_fields:
    #   tmp = input_dict[fea_name][1] % 1000  # 000000
    #   fea_dict[fea_name] = tf.RaggedTensor.from_row_lengths(
    #       tmp, input_dict[fea_name][0])
    # for fc in self._feature_configs:
    #   if fc.feature_type == fc.IdFeature or fc.feature_type == fc.TagFeature:
    #     input_0 = fc.input_names[0]
    #     fea_name = fc.feature_name if fc.HasField('feature_name') else input_0
    #     if not self._has_ev:
    #       tmp = input_dict[input_0][1] % fc.num_buckets
    #     else:
    #       tmp = input_dict[input_0][1]
    #     fea_dict[fea_name] = tf.RaggedTensor.from_row_lengths(tmp, input_dict[input_0][0])
    if self._has_ev:
      fea_dict['feature'] = tf.RaggedTensor.from_row_lengths(input_dict['feature'][1],
          input_dict['feature'][0])
    else:
      fea_dict['feature'] = tf.RaggedTensor.from_row_lengths(input_dict['feature'][1] % self._feature_configs[0].num_buckets,
          input_dict['feature'][0])

    lbl_dict = {}
    for lbl_name in self._label_fields:
      lbl_dict[lbl_name] = input_dict[lbl_name]
    return {'feature': fea_dict, 'label': lbl_dict}

  def _build(self, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN and self._data_config.num_epochs > 1:
      logging.info('will repeat train data for %d epochs' % self._data_config.num_epochs)
      my_files = self._my_files * self._data_config.num_epochs
    else:
      my_files = self._my_files
    for input_file in my_files:
      self._file_que.put(input_file)
    # add end signal
    for proc in self._proc_arr:
      self._file_que.put(None)
    logging.info('add input_files to file_que, qsize=%d' % self._file_que.qsize())

    out_types = {}
    out_shapes = {}
    for k in self._label_fields:
      out_types[k] = tf.int32
      out_shapes[k] = tf.TensorShape([None])
    # for k in self._effective_fields:
    #   out_types[k] = (tf.int64, tf.int32)
    #   out_shapes[k] = (tf.TensorShape([None]), tf.TensorShape([None]))
    out_types['feature'] = (tf.int64, tf.int32)
    out_shapes['feature'] = (tf.TensorShape([None]), tf.TensorShape([None]))

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
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    # dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    return dataset
