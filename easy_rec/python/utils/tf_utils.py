# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Common functions used for odps input."""
import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def get_tf_type(field_type):
  type_map = {
      DatasetConfig.INT32: tf.int32,
      DatasetConfig.INT64: tf.int64,
      DatasetConfig.STRING: tf.string,
      DatasetConfig.BOOL: tf.bool,
      DatasetConfig.FLOAT: tf.float32,
      DatasetConfig.DOUBLE: tf.double
  }
  assert field_type in type_map, 'invalid type: %s' % field_type
  return type_map[field_type]


def get_col_type(tf_type):
  type_map = {
      tf.int32: 'BIGINT',
      tf.int64: 'BIGINT',
      tf.string: 'STRING',
      tf.float32: 'FLOAT',
      tf.double: 'DOUBLE',
      tf.bool: 'BOOLEAN'
  }
  assert tf_type in type_map, 'invalid type: %s' % tf_type
  return type_map[tf_type]


def get_config_type(tf_type):
  type_map = {
      tf.int32: DatasetConfig.INT32,
      tf.int64: DatasetConfig.INT64,
      tf.string: DatasetConfig.STRING,
      tf.bool: DatasetConfig.BOOL,
      tf.float32: DatasetConfig.FLOAT,
      tf.double: DatasetConfig.DOUBLE
  }
  assert tf_type in type_map, 'invalid type: %s' % tf_type
  return type_map[tf_type]


def add_op(inputs):
  if not isinstance(inputs, list):
    return inputs
  if len(inputs) == 1:
    if isinstance(inputs[0], list):
      return tf.keras.layers.Add()(inputs[0])
    return inputs[0]
  return tf.keras.layers.Add()(inputs)


def dot_op(features):
  """Compute inner dot between any two pair tensors.

  Args:
    features:
    - List of 2D tensor with shape: ``(batch_size,embedding_size)``.
    - Or a 3D tensor with shape: ``(batch_size,field_size,embedding_size)``
  Return:
    - 2D tensor with shape: ``(batch_size, 1)``.
  """
  if isinstance(features, (list, tuple)):
    features = tf.stack(features, axis=1)
  assert features.shape.ndims == 3, 'input of dot func must be a 3D tensor or a list of 2D tensors'

  batch_size = tf.shape(features)[0]
  matrixdot = tf.matmul(features, features, transpose_b=True)
  feature_dim = matrixdot.shape[-1]

  ones_mat = tf.ones_like(matrixdot)
  lower_tri_mat = ones_mat - tf.linalg.band_part(ones_mat, 0, -1)
  lower_tri_mask = tf.cast(lower_tri_mat, tf.bool)
  result = tf.boolean_mask(matrixdot, lower_tri_mask)
  output_dim = feature_dim * (feature_dim - 1) // 2
  return tf.reshape(result, (batch_size, output_dim))
