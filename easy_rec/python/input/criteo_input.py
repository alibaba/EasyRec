# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf

from easy_rec.python.input.criteo_binary_reader import BinaryDataset
from easy_rec.python.input.input import Input

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CriteoInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1):
    super(CriteoInput, self).__init__(data_config, feature_config, input_path,
                                      task_index, task_num)
    if input_path:
      self._binary_reader = BinaryDataset(
          input_path.label_input,
          input_path.dense_input,
          input_path.category_input,
          self._batch_size,
          prefetch=self._prefetch_size,
          global_rank=self._task_index,
          global_size=self._task_num)
    else:
      self._binary_reader = None

  def _sample_generator(self):
    num_epoch = 0
    while num_epoch < self.num_epochs:
      logging.info('start epoch: %d' % num_epoch)
      for dense, category, labels in self._binary_reader:
        category = np.array(category, dtype=np.int64)
        yield dense, category, labels.reshape([-1])
      logging.info('finish epoch: %d' % num_epoch)
      num_epoch += 1

  def _to_fea_dict(self, dense, category, labels):
    field_dict = {}
    for fid in range(1, 14):
      fea_name = 'f%d' % fid
      field_dict[fea_name] = dense[:, fid - 1]

    for cid in range(1, 27):
      fea_name = 'c%d' % cid
      field_dict[fea_name] = category[:, cid - 1]
    field_dict['label'] = labels
    return field_dict

  def _build(self, mode, params):
    dataset = tf.data.Dataset.from_generator(
        self._sample_generator,
        output_types=(tf.float32, tf.int32, tf.int32),
        output_shapes=(tf.TensorShape([None, 13]), tf.TensorShape([None, 26]),
                       tf.TensorShape([None])))
    num_parallel_calls = self._data_config.num_parallel_calls
    dataset = dataset.map(
        self._to_fea_dict, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    dataset = dataset.map(
        map_func=self._preprocess, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
