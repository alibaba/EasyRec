# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Define cv_input, the base class for cv tasks."""

import os
import unittest

import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.input.csv_input import CSVInput
from easy_rec.python.input.csv_input_ex import CSVInputEx
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils import constant
from easy_rec.python.utils.test_utils import RunAsSubprocess

if tf.__version__ >= '2.0':
  from tensorflow.python.framework.ops import disable_eager_execution

  disable_eager_execution()
  tf = tf.compat.v1


class CSVInputTest(tf.test.TestCase):

  def __init__(self, methodName='CSVInputTest'):
    super(CSVInputTest, self).__init__(methodName=methodName)
    self._input_path = 'data/test/test.csv'
    self._input_path_with_quote = 'data/test/test_with_quote.csv'

  @RunAsSubprocess
  def test_csv_data(self):
    data_config_str = """
      input_fields {
        input_name: 'label'
        input_type: FLOAT
      }
      input_fields {
        input_name: 'field[1-3]'
        input_type: STRING
      }
      label_fields: 'label'
      batch_size: 1024
      num_epochs: 10000
      prefetch_size: 32
      auto_expand_input_fields: true
    """
    feature_config_str = """
      input_names: 'field1'
      shared_names: 'field[2-3]'
      feature_type: IdFeature
      embedding_dim: 32
      hash_bucket_size: 2000
    """
    dataset_config = DatasetConfig()
    text_format.Merge(data_config_str, dataset_config)
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)
    feature_configs = [feature_config]
    empty_config = FeatureConfig()
    empty_config.CopyFrom(feature_config)
    while len(empty_config.input_names) > 0:
      empty_config.input_names.pop()
    while len(empty_config.shared_names) > 0:
      empty_config.shared_names.pop()
    for input_name in feature_config.shared_names:
      input_names = config_util.auto_expand_names(input_name)
      for tmp_name in input_names:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(tmp_name)
        feature_configs.append(tmp_config)
    train_input_fn = CSVInput(dataset_config, feature_configs,
                              self._input_path).create_input()
    dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()
    init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)
    with self.test_session(config=session_config) as sess:
      sess.run(init_op)
      feature_dict, label_dict = sess.run([features, labels])

  @RunAsSubprocess
  def test_csv_data_flt_to_str_exception(self):
    data_config_str = """
      input_fields {
        input_name: 'label'
        input_type: FLOAT
      }
      input_fields {
        input_name: 'field1'
        input_type: STRING
      }
      input_fields {
        input_name: 'field[2-3]'
        input_type: FLOAT
      }
      label_fields: 'label'
      batch_size: 1024
      num_epochs: 10000
      prefetch_size: 32
      auto_expand_input_fields: true
    """
    feature_config_str = """
      input_names: 'field1'
      shared_names: 'field[2-3]'
      feature_type: IdFeature
      embedding_dim: 32
      hash_bucket_size: 2000
    """

    dataset_config = DatasetConfig()
    text_format.Merge(data_config_str, dataset_config)
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)
    feature_configs = [feature_config]
    empty_config = FeatureConfig()
    empty_config.CopyFrom(feature_config)
    while len(empty_config.input_names) > 0:
      empty_config.input_names.pop()
    while len(empty_config.shared_names) > 0:
      empty_config.shared_names.pop()
    for input_name in feature_config.shared_names:
      input_names = config_util.auto_expand_names(input_name)
      for tmp_name in input_names:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(tmp_name)
        feature_configs.append(tmp_config)
    train_input_fn = CSVInput(dataset_config, feature_configs,
                              self._input_path).create_input()
    try:
      dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)  # noqa: F841
      passed = True
    except Exception:
      passed = False
    assert not passed, 'if precision is not set, exception should be reported in convert float to string'

  @RunAsSubprocess
  def test_csv_data_flt_to_str(self):
    data_config_str = """
      input_fields {
        input_name: 'label'
        input_type: FLOAT
      }
      input_fields {
        input_name: 'field1'
        input_type: STRING
      }
      input_fields {
        input_name: 'field[2-3]'
        input_type: FLOAT
      }
      label_fields: 'label'
      batch_size: 1024
      num_epochs: 10000
      prefetch_size: 32
      auto_expand_input_fields: true
    """
    feature_config_str = """
      input_names: 'field1'
      shared_names: 'field[2-3]'
      feature_type: IdFeature
      embedding_dim: 32
      hash_bucket_size: 2000
      precision: 3
    """

    dataset_config = DatasetConfig()
    text_format.Merge(data_config_str, dataset_config)
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)
    feature_configs = [feature_config]
    empty_config = FeatureConfig()
    empty_config.CopyFrom(feature_config)
    while len(empty_config.input_names) > 0:
      empty_config.input_names.pop()
    while len(empty_config.shared_names) > 0:
      empty_config.shared_names.pop()
    for input_name in feature_config.shared_names:
      input_names = config_util.auto_expand_names(input_name)
      for tmp_name in input_names:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(tmp_name)
        feature_configs.append(tmp_config)
    train_input_fn = CSVInput(dataset_config, feature_configs,
                              self._input_path).create_input()

    dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()
    init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)
    with self.test_session(config=session_config) as sess:
      sess.run(init_op)
      feature_dict, label_dict = sess.run([features, labels])

  @RunAsSubprocess
  def test_csv_input_ex(self):
    data_config_str = """
      input_fields {
        input_name: 'label'
        input_type: FLOAT
      }
      input_fields {
        input_name: 'field[1-3]'
        input_type: STRING
      }
      label_fields: 'label'
      batch_size: 1024
      num_epochs: 10000
      prefetch_size: 32
      auto_expand_input_fields: true
    """
    feature_config_str = """
      input_names: 'field1'
      shared_names: 'field[2-3]'
      feature_type: IdFeature
      embedding_dim: 32
      hash_bucket_size: 2000
    """
    dataset_config = DatasetConfig()
    text_format.Merge(data_config_str, dataset_config)
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)
    feature_configs = [feature_config]
    empty_config = FeatureConfig()
    empty_config.CopyFrom(feature_config)
    while len(empty_config.input_names) > 0:
      empty_config.input_names.pop()
    while len(empty_config.shared_names) > 0:
      empty_config.shared_names.pop()
    for input_name in feature_config.shared_names:
      input_names = config_util.auto_expand_names(input_name)
      for tmp_name in input_names:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(tmp_name)
        feature_configs.append(tmp_config)
    train_input_fn = CSVInputEx(dataset_config, feature_configs,
                                self._input_path_with_quote).create_input()
    dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()
    init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)
    with self.test_session(config=session_config) as sess:
      sess.run(init_op)
      feature_dict, label_dict = sess.run([features, labels])

  @unittest.skipIf('AVX_TEST' not in os.environ,
                   'Only execute when avx512 instructions are supported')
  @RunAsSubprocess
  def test_csv_input_ex_avx(self):
    constant.enable_avx_str_split()
    self.test_csv_input_ex()
    constant.disable_avx_str_split()

  @RunAsSubprocess
  def test_csv_data_ignore_error(self):
    data_config_str = """
      input_fields {
        input_name: 'label'
        input_type: FLOAT
      }
      input_fields {
        input_name: 'field[1-3]'
        input_type: STRING
      }
      label_fields: 'label'
      batch_size: 32
      num_epochs: 10000
      prefetch_size: 32
      auto_expand_input_fields: true
      ignore_error: true
    """
    feature_config_str = """
      input_names: 'field1'
      shared_names: 'field[2-3]'
      feature_type: IdFeature
      embedding_dim: 32
      hash_bucket_size: 2000
    """
    dataset_config = DatasetConfig()
    text_format.Merge(data_config_str, dataset_config)
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)
    feature_configs = [feature_config]
    empty_config = FeatureConfig()
    empty_config.CopyFrom(feature_config)
    while len(empty_config.input_names) > 0:
      empty_config.input_names.pop()
    while len(empty_config.shared_names) > 0:
      empty_config.shared_names.pop()
    for input_name in feature_config.shared_names:
      input_names = config_util.auto_expand_names(input_name)
      for tmp_name in input_names:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(tmp_name)
        feature_configs.append(tmp_config)
    train_input_fn = CSVInput(dataset_config, feature_configs,
                              self._input_path_with_quote).create_input()
    dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()
    init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)
    with self.test_session(config=session_config) as sess:
      sess.run(init_op)
      feature_dict, label_dict = sess.run([features, labels])


if __name__ == '__main__':
  tf.test.main()
