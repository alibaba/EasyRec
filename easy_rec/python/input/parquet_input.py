# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import multiprocessing
import queue

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.input.input import Input

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
               pipeline_config=None):
    super(ParquetInput,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    if input_path is None:
      return

    self._input_files = []
    for sub_path in input_path.strip().split(','):
      self._input_files.extend(gfile.Glob(sub_path))
    logging.info('parquet input_path=%s file_num=%d' %
                 (input_path, len(self._input_files)))
    self._data_que = multiprocessing.Queue(
        maxsize=self._data_config.prefetch_size)
    self._file_que = multiprocessing.Queue(maxsize=len(self._input_files))
    for input_file in self._input_files:
      self._file_que.put(input_file)

    file_num = len(self._input_files)
    num_proc = 8
    if file_num < num_proc:
      num_proc = file_num
    logging.info('file_num=%d num_proc=%d' % (file_num, num_proc))
    self._proc_arr = []
    for proc_id in range(num_proc):
      proc = multiprocessing.Process(
          target=self._parse_one_file, args=(proc_id,))
      self._proc_arr.append(proc)

  def _parse_one_file(self, proc_id):
    all_fields = list(self._label_fields) + list(self._effective_fields)
    logging.info('data proc %d start' % proc_id)
    num_files = 0
    while True:
      try:
        input_file = self._file_que.get(block=False)
      except queue.Empty:
        break
      num_files += 1
      input_data = pd.read_parquet(input_file, columns=all_fields)
      data_len = len(input_data[all_fields[0]])
      batch_num = int(data_len / self._batch_size)
      res_num = data_len % self._batch_size
      logging.info(
          'proc[%d] read file %s sample_num=%d batch_num=%d res_num=%d' %
          (proc_id, input_file, data_len, batch_num, res_num))
      sid = 0
      for batch_id in range(batch_num):
        eid = sid + self._batch_size
        data_dict = {}
        for k in self._label_fields:
          data_dict[k] = np.array([x[0] for x in input_data[k][sid:eid]],
                                  dtype=np.float32)
        for k in self._effective_fields:
          val = input_data[k][sid:eid]
          all_lens = np.array([len(x) for x in val], dtype=np.int32)
          all_vals = np.concatenate(list(val))
          assert np.sum(all_lens) == len(
              all_vals), 'len(all_vals)=%d np.sum(all_lens)=%d' % (
                  len(all_vals), np.sum(all_lens))
          data_dict[k] = (all_lens, all_vals)
        self._data_que.put(data_dict)
        sid += self._batch_size
      if res_num > 0:
        logging.info('proc[%d] add final sample' % proc_id)
        data_dict = {}
        for k in self._label_fields:
          data_dict[k] = np.array([x[0] for x in input_data[k][sid:]],
                                  dtype=np.float32)
        for k in self._effective_fields:
          val = input_data[k][sid:]
          all_lens = np.array([len(x) for x in val], dtype=np.int32)
          all_vals = np.concatenate(list(val))
          data_dict[k] = (all_lens, all_vals)
        self._data_que.put(data_dict)
    self._data_que.put(None)
    logging.info('data proc %d done, file_num=%d' % (proc_id, num_files))

  def _sample_generator(self):
    for proc in self._proc_arr:
      proc.start()
    done_proc_cnt = 0
    fetch_timeout_cnt = 0
    while True:
      try:
        sample = self._data_que.get(block=False)
        if sample is None:
          done_proc_cnt += 1
        else:
          yield sample
      except queue.Empty:
        fetch_timeout_cnt += 1
        if done_proc_cnt >= len(self._proc_arr):
          logging.info('all sample finished, fetch_timeout_cnt=%d' %
                       fetch_timeout_cnt)
          break
    for proc in self._proc_arr:
      proc.join()

  def _to_fea_dict(self, input_dict):
    fea_dict = {}
    # for fea_name in self._effective_fields:
    #   tmp = input_dict[fea_name][1] % 1000  # 000000
    #   fea_dict[fea_name] = tf.RaggedTensor.from_row_lengths(
    #       tmp, input_dict[fea_name][0])
    for fc in self._feature_configs:
      if fc.feature_type == fc.IdFeature or fc.feature_type == fc.TagFeature:
        input_0 = fc.input_names[0]
        fea_name = fc.feature_name if fc.HasField('feature_name') else input_0
        tmp = input_dict[input_0][1] % fc.num_buckets
        fea_dict[fea_name] = tf.RaggedTensor.from_row_lengths(tmp, input_dict[input_0][0])

    lbl_dict = {}
    for lbl_name in self._label_fields:
      lbl_dict[lbl_name] = input_dict[lbl_name]
    return {'feature': fea_dict, 'label': lbl_dict}

  def _build(self, mode, params):
    out_types = {}
    out_shapes = {}
    for k in self._label_fields:
      out_types[k] = tf.int32
      out_shapes[k] = tf.TensorShape([None])
    for k in self._effective_fields:
      out_types[k] = (tf.int64, tf.int32)
      out_shapes[k] = (tf.TensorShape([None]), tf.TensorShape([None]))

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
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
