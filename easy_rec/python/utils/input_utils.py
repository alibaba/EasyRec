# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def get_type_defaults(field_type, default_val=''):
  type_defaults = {
      DatasetConfig.INT32: 0,
      DatasetConfig.INT64: 0,
      DatasetConfig.STRING: '',
      DatasetConfig.BOOL: False,
      DatasetConfig.FLOAT: 0.0,
      DatasetConfig.DOUBLE: 0.0
  }
  assert field_type in type_defaults, 'invalid type: %s' % field_type
  if default_val == '':
    default_val = type_defaults[field_type]
  if field_type == DatasetConfig.INT32:
    return int(default_val)
  elif field_type == DatasetConfig.INT64:
    return np.int64(default_val)
  elif field_type == DatasetConfig.STRING:
    return default_val
  elif field_type == DatasetConfig.BOOL:
    return default_val.lower() == 'true'
  elif field_type in [DatasetConfig.FLOAT]:
    return float(default_val)
  elif field_type in [DatasetConfig.DOUBLE]:
    return np.float64(default_val)

  return type_defaults[field_type]


def string_to_number(field, ftype, default_value, name=''):
  """Type conversion for parsing rtp fg input format.

  Args:
    field: field to be converted.
    ftype: field dtype set in DatasetConfig.
    default_value: default value for this field
    name: field name for
  Returns: A name for the operation (optional).
  """
  default_vals = tf.tile(tf.constant([str(default_value)]), tf.shape(field))
  field = tf.where(tf.greater(tf.strings.length(field), 0), field, default_vals)

  if ftype in [DatasetConfig.INT32, DatasetConfig.INT64]:
    # Int type is not supported in fg.
    # If you specify INT32, INT64 in DatasetConfig, you need to perform a cast at here.
    tmp_field = tf.string_to_number(
        field, tf.double, name='field_as_flt_%s' % name)
    if ftype in [DatasetConfig.INT64]:
      tmp_field = tf.cast(tmp_field, tf.int64)
    else:
      tmp_field = tf.cast(tmp_field, tf.int32)
  elif ftype in [DatasetConfig.FLOAT]:
    tmp_field = tf.string_to_number(
        field, tf.float32, name='field_as_flt_%s' % name)
  elif ftype in [DatasetConfig.DOUBLE]:
    tmp_field = tf.string_to_number(
        field, tf.float64, name='field_as_flt_%s' % name)
  elif ftype in [DatasetConfig.BOOL]:
    tmp_field = tf.logical_or(tf.equal(field, 'True'), tf.equal(field, 'true'))
  elif ftype in [DatasetConfig.STRING]:
    tmp_field = field
  else:
    assert False, 'invalid types: %s' % str(ftype)
  return tmp_field


def _calculate_concat_shape(shapes):
  for shape in shapes:
    assert len(shape.get_shape()) == 1
  shapes_stack = tf.stack(shapes, axis=0)
  batch_size = tf.reduce_sum(shapes_stack[:,:1], axis=0)
  other_sizes = tf.reduce_max(shapes_stack[:,1:], axis=0)
  return tf.cond(
    tf.equal(
      tf.shape(other_sizes, out_type=tf.int32)[0],
      tf.constant(0, dtype=tf.int32)),
    lambda: batch_size,
    lambda: tf.concat([batch_size, other_sizes], axis=0)
  )


def _accumulate_concat_indices(indices_list, shape_list):
  with tf.name_scope('accumulate_concat_indices'):
    assert len(indices_list) != 0
    indices_shape = indices_list[0].get_shape()
    assert len(indices_shape) == 2
    rank = indices_shape[1].value
    assert rank is not None and rank > 0
    indices_0_list = [indices_list[0][:,:1]]
    offset = shape_list[0][0]
    for i in range(1, len(indices_list)):
      indices_0_list.append(tf.add(indices_list[i][:,:1], offset))
      if i == len(indices_list) - 1:
        break
      offset = tf.add(offset, shape_list[i][0])
    if rank == 1:
      return indices_0_list
    else:
      return [
        tf.concat([indices_0, indices[:,1:]], axis=1)
        for indices_0, indices
        in zip(indices_0_list, indices_list)
      ]


def _dense_to_sparse(dense_tensor):
  with tf.name_scope('dense_to_sparse'):
    shape = tf.shape(dense_tensor, out_type=tf.int64, name='sparse_shape')
    nelems = tf.size(dense_tensor, out_type=tf.int64, name='num_elements')
    indices = tf.transpose(
      tf.unravel_index(tf.range(nelems, dtype=tf.int64), shape),
      name='sparse_indices')
    values = tf.reshape(dense_tensor, [nelems], name='sparse_values')
    return tf.SparseTensor(indices, values, shape)


def _concat_parsed_features_impl(features, name):
  is_sparse = False
  for feature in features:
    if isinstance(feature, tf.SparseTensor):
      is_sparse = True
      break
  feature_ranks = [len(feature.get_shape()) for feature in features]
  max_rank = max(feature_ranks)
  if is_sparse:
    concat_indices = []
    concat_values = []
    concat_shapes = []
    # concat_tensors = []
    for i in range(len(features)):
      with tf.name_scope('sparse_preprocess_{}'.format(i)):
        feature = features[i]
        if isinstance(feature, tf.Tensor):
          feature = _dense_to_sparse(feature)
        feature_rank = feature_ranks[i]
        if feature_rank < max_rank:
          # expand dimensions
          feature = tf.SparseTensor(
            tf.pad(feature.indices,
                   [[0,0], [0,max_rank-feature_rank]],
                   constant_values=0,
                   name='indices_expanded'),
            feature.values,
            tf.pad(feature.dense_shape,
                   [[0,max_rank-feature_rank]],
                   constant_values=1,
                   name='shape_expanded')
          )
        concat_indices.append(feature.indices)
        concat_values.append(feature.values)
        concat_shapes.append(feature.dense_shape)
    with tf.name_scope('sparse_indices'):
      concat_indices = _accumulate_concat_indices(concat_indices, concat_shapes)
      sparse_indices = tf.concat(concat_indices, axis=0)
    with tf.name_scope('sparse_values'):
      sparse_values = tf.concat(concat_values, axis=0)
    with tf.name_scope('sparse_shape'):
      sparse_shape = _calculate_concat_shape(concat_shapes)
    return tf.SparseTensor(
      sparse_indices,
      sparse_values,
      sparse_shape
    )
  else:
    # expand dimensions
    for i in range(len(features)):
      with tf.name_scope('dense_preprocess_{}'.format(i)):
        feature_rank = feature_ranks[i]
        if feature_rank < max_rank:
          new_shape = tf.pad(tf.shape(feature),
                             [[0, max_rank - feature_rank]],
                             constant_values=1,
                             name='shape_expanded')
          features[i] = tf.reshape(features[i], new_shape, name='dense_expanded')
    # assumes that dense tensors are of the same shape
    return tf.concat(features, axis=0, name='dense_concat')


def concat_parsed_features(features, name=None):
  if name:
    with tf.name_scope('concat_parsed_features__{}'.format(name)):
      return _concat_parsed_features_impl(features, name)
  else:
    return _concat_parsed_features_impl(features, '<unknown>')
