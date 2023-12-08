# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.input.input import Input
from easy_rec.python.utils.input_utils import get_type_defaults

try:
  from tensorflow.python.data.experimental.ops import parquet_dataset_ops
  from tensorflow.python.data.experimental.ops import parquet_pybind
  from tensorflow.python.data.experimental.ops import dataframe
  from tensorflow.python.ops import gen_ragged_conversion_ops
  from tensorflow.python.ops.work_queue import WorkQueue
  _has_deep_rec = True
except Exception:
  _has_deep_rec = False
  pass


class ParquetInputV3(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None,
               **kwargs):
    if not _has_deep_rec:
      raise RuntimeError('You should install DeepRec first.')
    super(ParquetInputV3,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

    self._ignore_val_dict = {}
    for f in data_config.input_fields:
      if f.HasField('ignore_val'):
        self._ignore_val_dict[f.input_name] = get_type_defaults(
            f.input_type, f.ignore_val)

    self._true_type_dict = {}
    for fc in self._feature_configs:
      if fc.feature_type in [fc.IdFeature, fc.TagFeature, fc.SequenceFeature]:
        if fc.hash_bucket_size > 0 or len(
            fc.vocab_list) > 0 or fc.HasField('vocab_file'):
          self._true_type_dict[fc.input_names[0]] = tf.string
        else:
          self._true_type_dict[fc.input_names[0]] = tf.int64
        if len(fc.input_names) > 1:
          self._true_type_dict[fc.input_names[1]] = tf.float32
      if fc.feature_type == fc.RawFeature:
        self._true_type_dict[fc.input_names[0]] = tf.float32

    self._reserve_fields = None
    self._reserve_types = None
    if 'reserve_fields' in kwargs and 'reserve_types' in kwargs:
      self._reserve_fields = kwargs['reserve_fields']
      self._reserve_types = kwargs['reserve_types']

    # In ParquetDataset multi_value use input type
    self._multi_value_types = {}

  def _ignore_and_cast(self, name, value):
    ignore_value = self._ignore_val_dict.get(name, None)
    if ignore_value:
      if isinstance(value, tf.SparseTensor):
        indices = tf.where(tf.equal(value.values, ignore_value))
        value = tf.SparseTensor(
            tf.gather_nd(value.indices, indices),
            tf.gather_nd(value.values, indices), value.dense_shape)
      elif isinstance(value, tf.Tensor):
        indices = tf.where(tf.not_equal(value, ignore_value), name='indices')
        value = tf.SparseTensor(
            indices=indices,
            values=tf.gather_nd(value, indices),
            dense_shape=tf.shape(value, out_type=tf.int64))
    dtype = self._true_type_dict.get(name, None)
    if dtype:
      value = tf.cast(value, dtype)
    return value

  def _parse_dataframe_value(self, value):
    if len(value.nested_row_splits) == 0:
      return value.values
    value.values.set_shape([None])
    sparse_value = gen_ragged_conversion_ops.ragged_tensor_to_sparse(
        value.nested_row_splits, value.values)
    return tf.SparseTensor(sparse_value.sparse_indices,
                           sparse_value.sparse_values,
                           sparse_value.sparse_dense_shape)

  def _parse_dataframe(self, df):
    inputs = {}
    for k, v in df.items():
      if k in self._effective_fields:
        if isinstance(v, dataframe.DataFrame.Value):
          v = self._parse_dataframe_value(v)
      elif k in self._label_fields:
        if isinstance(v, dataframe.DataFrame.Value):
          v = v.values
      elif k in self._reserve_fields:
        if isinstance(v, dataframe.DataFrame.Value):
          v = v.values
      else:
        continue
      inputs[k] = v
    return inputs

  def _build(self, mode, params):
    input_files = []
    for sub_path in self._input_path.strip().split(','):
      input_files.extend(gfile.Glob(sub_path))
    file_num = len(input_files)
    logging.info('[task_index=%d] total_file_num=%d task_num=%d' %
                 (self._task_index, file_num, self._task_num))

    task_index = self._task_index
    task_num = self._task_num
    if self._data_config.chief_redundant:
      task_index = max(self._task_index - 1, 0)
      task_num = max(self._task_num - 1, 1)

    if self._data_config.pai_worker_queue and \
        mode == tf.estimator.ModeKeys.TRAIN:
      work_queue = WorkQueue(
          input_files,
          num_epochs=self.num_epochs,
          shuffle=self._data_config.shuffle)
      my_files = work_queue.input_dataset()
    else:
      my_files = []
      for file_id in range(file_num):
        if (file_id % task_num) == task_index:
          my_files.append(input_files[file_id])

    parquet_fields = parquet_pybind.parquet_fields(input_files[0])
    parquet_input_fields = []
    for f in parquet_fields:
      if f.name in self._input_fields:
        parquet_input_fields.append(f)

    all_fields = set(self._effective_fields)
    if mode != tf.estimator.ModeKeys.PREDICT:
      all_fields |= set(self._label_fields)
    if self._reserve_fields:
      all_fields |= set(self._reserve_fields)

    selected_fields = []
    for f in parquet_input_fields:
      if f.name in all_fields:
        selected_fields.append(f)

    num_parallel_reads = min(self._data_config.num_parallel_calls,
                             len(input_files) // task_num)
    dataset = parquet_dataset_ops.ParquetDataset(
        my_files,
        batch_size=self._batch_size,
        fields=selected_fields,
        drop_remainder=self._data_config.drop_remainder,
        num_parallel_reads=num_parallel_reads)
    # partition_count=task_num,
    # partition_index=task_index)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if self._data_config.shuffle:
        dataset = dataset.shuffle(
            self._data_config.shuffle_buffer_size,
            seed=2020,
            reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.map(
        self._parse_dataframe,
        num_parallel_calls=self._data_config.num_parallel_calls)

    # preprocess is necessary to transform data
    # so that they could be feed into FeatureColumns
    dataset = dataset.map(
        map_func=self._preprocess,
        num_parallel_calls=self._data_config.num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset

  def _preprocess(self, field_dict):
    for k, v in field_dict.items():
      field_dict[k] = self._ignore_and_cast(k, v)
    return super(ParquetInputV3, self)._preprocess(field_dict)
