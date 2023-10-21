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
    self._data_que = multiprocessing.Queue(
        maxsize=self._data_config.prefetch_size)


    file_num = len(self._input_files)
    logging.info('[task_index=%d] total_file_num=%d task_num=%d' % (task_index, file_num, task_num))
    avg_file_num = int(file_num / task_num)
    res_file_num = file_num % task_num
    file_sid = task_index * avg_file_num + min(res_file_num, task_index)
    file_eid = (task_index + 1) * avg_file_num + min(res_file_num, task_index+1)
    self._my_files = self._input_files[file_sid:file_eid] 

    logging.info('[task_index=%d] task_file_num=%d' % (task_index, len(self._my_files)))
    self._file_que = multiprocessing.Queue()

    num_proc = 8
    if file_num < num_proc:
      num_proc = file_num

    self._proc_start = False
    self._proc_arr = []
    for proc_id in range(num_proc):
      proc = multiprocessing.Process(
          target=self._load_data_proc, args=(proc_id,))
      self._proc_arr.append(proc)

  def _load_data_proc(self, proc_id):
    all_fields = list(self._label_fields) + list(self._effective_fields)
    logging.info('data proc %d start' % proc_id)
    num_files = 0
    part_data_dict = {}
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
        accum_res_num = 0
        data_dict = {}
        part_data_dict_n = {}
        for k in self._label_fields:
          tmp_lbls = np.array([x[0] for x in input_data[k][sid:]],
                                  dtype=np.float32)
          if part_data_dict is not None and k in part_data_dict:
            tmp_lbls = np.concatenate([part_data_dict[k], tmp_lbls], axis=0)
            if len(tmp_lbls) > self._batch_size:
              data_dict[k] = tmp_lbls[:self._batch_size] 
              part_data_dict_n[k] = tmp_lbls[self._batch_size:]
            elif len(tmp_lbls) == self._batch_size:
              data_dict[k] = tmp_lbls
            else:
              part_data_dict_n[k] = tmp_lbls
          else:
            part_data_dict_n[k] = tmp_lbls
        for k in self._effective_fields:
          val = input_data[k][sid:]
          all_lens = np.array([len(x) for x in val], dtype=np.int32)
          all_vals = np.concatenate(list(val))
          if part_data_dict is not None and k in part_data_dict:
            tmp_lens = np.concatenate([part_data_dict[k][0], all_lens], axis=0)
            tmp_vals = np.concatenate([part_data_dict[k][1], all_vals], axis=0)
            if len(tmp_lens) > self._batch_size:
              tmp_res_lens = tmp_lens[self._batch_size:]
              tmp_lens = tmp_lens[:self._batch_size]
              tmp_num_elems = np.sum(tmp_lens)
              tmp_res_vals = tmp_vals[tmp_num_elems:]
              tmp_vals = tmp_vals[:tmp_num_elems]
              part_data_dict_n[k] = (tmp_res_lens, tmp_res_vals)
              data_dict[k] = (tmp_lens, tmp_vals)
            elif len(tmp_lens) == self._batch_size:
              data_dict[k] = (tmp_lens, tmp_vals)
            else:
              part_data_dict_n[k] = (tmp_lens, tmp_vals)
          else:
            part_data_dict_n[k] = (all_lens, all_vals)
        if len(data_dict) > 0:
          self._data_que.put(data_dict)
        part_data_dict = part_data_dict_n
    if len(part_data_dict) > 0:
      if not self._data_config.drop_remainder:
        self._data_que.put(part_data_dict)
      else:
        logging.warning('drop remain %d samples as drop_remainder is set' % \
             len(part_data_dict[self._label_fields[0]]))
    self._data_que.put(None)
    logging.info('data proc %d done, file_num=%d' % (proc_id, num_files))

  def _sample_generator(self):
    if not self._proc_start:
      self._proc_start = True 
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
        if not self._has_ev:
          tmp = input_dict[input_0][1] % fc.num_buckets
        else:
          tmp = input_dict[input_0][1]
        fea_dict[fea_name] = tf.RaggedTensor.from_row_lengths(tmp, input_dict[input_0][0])

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
    # dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
