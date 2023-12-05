# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.platform import gfile

from easy_rec.python.input.input import Input
from easy_rec.python.utils.input_utils import get_type_defaults
from easy_rec.python.utils.tf_utils import get_tf_type

try:
  from tensorflow.python.data.experimental.ops import parquet_dataset_ops
  from tensorflow.python.data.experimental.ops import parquet_pybind
  from tensorflow.python.data.experimental.ops import dataframe
  from tensorflow.python.ops import gen_ragged_conversion_ops
  from tensorflow.python.framework import sparse_tensor
except Exception:
  logging.error('You should install DeepRec first.')
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
    super(ParquetInputV3,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    if input_path is None:
      return

    self._input_files = []
    for sub_path in input_path.strip().split(','):
      self._input_files.extend(gfile.Glob(sub_path))

    file_num = len(self._input_files)
    logging.info('[task_index=%d] total_file_num=%d task_num=%d' %
                 (task_index, file_num, task_num))

    self._my_files = []
    for file_id in range(file_num):
      if (file_id % task_num) == task_index:
        self._my_files.append(self._input_files[file_id])

    parquet_fields = parquet_pybind.parquet_fields(self._input_files[0])
    self._parquet_input_fields = []
    self._tf_type_dict = {}
    for f in parquet_fields:
      if f.name in self._input_fields:
        self._parquet_input_fields.append(f)
    self._tf_type_dict = {}
    self._ignore_val_dict = {}
    for f in data_config.input_fields:
      self._tf_type_dict[f.input_name] = get_tf_type(f.input_type)
      if f.HasField('ignore_val'):
        self._ignore_val_dict[f.input_name] = get_type_defaults(
            f.input_type, f.ignore_val)
      else:
        self._ignore_val_dict[f.input_name] = None

    self._reserve_fields = None
    self._reserve_types = None
    if 'reserve_fields' in kwargs and 'reserve_types' in kwargs:
      self._reserve_fields = kwargs['reserve_fields']
      self._reserve_types = kwargs['reserve_types']

  def _cast_and_ignore(self, value, dtype, ignore_value):
    value = tf.cast(value, dtype)
    if ignore_value is not None:
      assert isinstance(value, sparse_tensor.SparseTensor
                        ), 'only SparseTensor support ignore_value now.'
      mask = tf.equal(value.values, ignore_value)
      value = sparse_tensor.SparseTensor(
          tf.boolean_mask(value.indices, mask),
          tf.boolean_mask(value.values, mask), value.dense_shape)
    return value

  def _parse_dataframe_value(self, value):
    if len(value.nested_row_splits) == 0:
      return value.values
    value.values.set_shape([None])
    sparse_value = gen_ragged_conversion_ops.ragged_tensor_to_sparse(
        value.nested_row_splits, value.values)
    return sparse_tensor.SparseTensor(sparse_value.sparse_indices,
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
      inputs[k] = self._cast_and_ignore(v, self._tf_type_dict[k],
                                        self._ignore_val_dict[k])
    return inputs

  def _build(self, mode, params):
    all_fields = set(self._effective_fields)
    if mode != tf.estimator.ModeKeys.PREDICT:
      all_fields |= set(self._label_fields)
    if self._reserve_fields:
      all_fields |= set(self._reserve_fields)

    selected_fields = []
    for f in self._parquet_input_fields:
      if f.name in all_fields:
        selected_fields.append(f)

    dataset = parquet_dataset_ops.ParquetDataset(
        self._my_files,
        batch_size=self._batch_size,
        fields=selected_fields,
        drop_remainder=self._data_config.drop_remainder)
    # partition_count=self._task_num,
    # partition_index=self._task_index)

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
