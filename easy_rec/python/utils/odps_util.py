# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Common functions used for odps input."""
from tensorflow.python.framework import dtypes

from easy_rec.python.protos.dataset_pb2 import DatasetConfig


def is_type_compatiable(odps_type, input_type):
  """Check that odps_type are compatiable with input_type."""
  type_map = {
      'bigint': DatasetConfig.INT64,
      'string': DatasetConfig.STRING,
      'double': DatasetConfig.DOUBLE
  }
  tmp_type = type_map[odps_type]
  if tmp_type == input_type:
    return True
  else:
    float_types = [DatasetConfig.FLOAT, DatasetConfig.DOUBLE]
    int_types = [DatasetConfig.INT32, DatasetConfig.INT64]
    if tmp_type in float_types and input_type in float_types:
      return True
    elif tmp_type in int_types and input_type in int_types:
      return True
    else:
      return False


def odps_type_to_input_type(odps_type):
  """Check that odps_type are compatiable with input_type."""
  odps_type_map = {
      'bigint': DatasetConfig.INT64,
      'string': DatasetConfig.STRING,
      'double': DatasetConfig.DOUBLE
  }
  assert odps_type in odps_type_map, 'only support [bigint, string, double]'
  input_type = odps_type_map[odps_type]
  return input_type


def check_input_field_and_types(data_config):
  """Check compatibility of input in data_config.

  check that data_config.input_fields are compatible with
  data_config.selected_cols and data_config.selected_types.

  Args:
    data_config: instance of DatasetConfig
  """
  input_fields = [x.input_name for x in data_config.input_fields]
  input_field_types = [x.input_type for x in data_config.input_fields]
  selected_cols = data_config.selected_cols if data_config.selected_cols else None
  selected_col_types = data_config.selected_col_types if data_config.selected_col_types else None
  if not selected_cols:
    return

  selected_cols = selected_cols.split(',')
  for x in input_fields:
    assert x in selected_cols, 'column %s is not in table' % x
  if selected_col_types:
    selected_types = selected_col_types.split(',')
    type_map = {x: y for x, y in zip(selected_cols, selected_types)}
    for x, y in zip(input_fields, input_field_types):
      tmp_type = type_map[x]
      assert is_type_compatiable(tmp_type, y), \
          'feature[%s] type error: odps %s is not compatible with input_type %s' % (
              x, tmp_type, DatasetConfig.FieldType.Name(y))


def odps_type_2_tf_type(odps_type):
  if odps_type == 'string':
    return dtypes.string
  elif odps_type == 'bigint':
    return dtypes.int64
  elif odps_type in ['double', 'float']:
    return dtypes.float32
  else:
    return dtypes.string
