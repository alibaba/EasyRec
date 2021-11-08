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


def string_to_number(record_default, cur_field, name=''):
  tmp_field = cur_field
  if type(record_default) in [int, np.int32, np.int64]:
    tmp_field = tf.string_to_number(
        cur_field, tf.double, name='field_as_int_%s' % name)
    if type(record_default) in [np.int64]:
      tmp_field = tf.cast(tmp_field, tf.int64)
    else:
      tmp_field = tf.cast(tmp_field, tf.int32)
  elif type(record_default) in [float, np.float32]:
    tmp_field = tf.string_to_number(
        cur_field, tf.float32, name='field_as_flt_%s' % name)
  elif type(record_default) in [np.float64]:
    tmp_field = tf.string_to_number(
        cur_field, tf.float64, name='field_as_flt_%s' % name)
  elif type(record_default) in [str, type(u''), bytes]:
    pass
  elif type(record_default) == bool:
    tmp_field = tf.logical_or(
        tf.equal(cur_field, 'True'), tf.equal(cur_field, 'true'))
  else:
    assert 'invalid types: %s' % str(type(record_default))
  return tmp_field
