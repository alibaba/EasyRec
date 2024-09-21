# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common util functions used by layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from google.protobuf import struct_pb2
from google.protobuf.descriptor import FieldDescriptor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import variables

try:
  from tensorflow.python.ops import kv_variable_ops
except ImportError:
  kv_variable_ops = None

ColumnNameInCollection = {}


def _tensor_to_map(tensor):
  return {
      'node_path': tensor.name,
      'shape': tensor.shape.as_list() if tensor.shape else None,
      'dtype': tensor.dtype.name
  }


def _tensor_to_tensorinfo(tensor):
  tensor_info = {}
  if isinstance(tensor, sparse_tensor.SparseTensor):
    tensor_info['is_dense'] = False
    tensor_info['values'] = _tensor_to_map(tensor.values)
    tensor_info['indices'] = _tensor_to_map(tensor.indices)
    tensor_info['dense_shape'] = _tensor_to_map(tensor.dense_shape)
  else:
    tensor_info['is_dense'] = True
    tensor_info.update(_tensor_to_map(tensor))
  return tensor_info


def add_tensor_to_collection(collection_name, name, tensor):
  tensor_info = _tensor_to_tensorinfo(tensor)
  tensor_info['name'] = name
  update_attr_to_collection(collection_name, tensor_info)


def append_tensor_to_collection(collection_name, name, key, tensor):
  tensor_info = _tensor_to_tensorinfo(tensor)
  append_attr_to_collection(collection_name, name, key, tensor_info)


def _collection_item_key(col, name):
  return '%d#%s' % (id(col), name)


def _process_item(collection_name, name, func):
  col = ops.get_collection_ref(collection_name)
  item_found = {}
  idx_found = -1

  # add id(col) because col may re-new sometimes
  key = _collection_item_key(col, name)
  if key in ColumnNameInCollection:
    idx_found = ColumnNameInCollection[key]
    if idx_found >= len(col):
      raise Exception(
          'Find column name in collection failed: index out of range')

    item_found = json.loads(col[idx_found])
    if item_found['name'] != name:
      raise Exception(
          'Find column name in collection failed: item name not match')
    func(item_found)
    col[idx_found] = json.dumps(item_found)
  else:
    func(item_found)
    col.append(json.dumps(item_found))
    ColumnNameInCollection[key] = len(col) - 1


def append_attr_to_collection(collection_name, name, key, value):

  def append(item_found):
    if key not in item_found:
      item_found[key] = []
    item_found[key].append(value)

  _process_item(collection_name, name, append)


def update_attr_to_collection(collection_name, attrs):

  def update(item_found):
    item_found.update(attrs)

  _process_item(collection_name, attrs['name'], update)


def unique_name_in_collection(collection_name, name):
  col = ops.get_collection_ref(collection_name)
  unique_name = name
  index = 0
  while True:
    key = _collection_item_key(col, unique_name)
    if key not in ColumnNameInCollection:
      break
    index += 1
    unique_name = '%s_%d' % (name, index)
  return unique_name


def gen_embedding_attrs(column=None,
                        variable=None,
                        bucket_size=None,
                        combiner=None,
                        is_embedding_var=None):
  attrs = dict()
  attrs['name'] = column.name
  attrs['bucket_size'] = bucket_size
  attrs['combiner'] = combiner
  attrs['is_embedding_var'] = is_embedding_var
  attrs['weights_op_path'] = variable.name
  if kv_variable_ops:
    if isinstance(variable, kv_variable_ops.EmbeddingVariable):
      attrs['is_embedding_var'] = True
      attrs['embedding_var_keys'] = variable._shared_name + '-keys'
      attrs['embedding_var_values'] = variable._shared_name + '-values'
    elif (isinstance(variable, variables.PartitionedVariable)) and \
        (isinstance(variable._get_variable_list()[0], kv_variable_ops.EmbeddingVariable)):
      attrs['embedding_var_keys'] = [v._shared_name + '-keys' for v in variable]
      attrs['embedding_var_values'] = [
          v._shared_name + '-values' for v in variable
      ]
    else:
      attrs['is_embedding_var'] = False
  else:
    attrs['is_embedding_var'] = False
  return attrs


def mark_input_src(name, src_desc):
  ops.add_to_collection(ops.GraphKeys.RANK_SERVICE_INPUT_SRC,
                        json.dumps({
                            'name': name,
                            'src': src_desc
                        }))


def is_proto_message(pb_obj, field):
  if not hasattr(pb_obj, 'DESCRIPTOR'):
    return False
  if field not in pb_obj.DESCRIPTOR.fields_by_name:
    return False
  field_type = pb_obj.DESCRIPTOR.fields_by_name[field].type
  return field_type == FieldDescriptor.TYPE_MESSAGE


class Parameter(object):

  def __init__(self, params, is_struct, l2_reg=None):
    self.params = params
    self.is_struct = is_struct
    self._l2_reg = l2_reg

  @staticmethod
  def make_from_pb(config):
    return Parameter(config, False)

  def get_pb_config(self):
    assert not self.is_struct, 'Struct parameter can not convert to pb config'
    return self.params

  @property
  def l2_regularizer(self):
    return self._l2_reg

  @l2_regularizer.setter
  def l2_regularizer(self, value):
    self._l2_reg = value

  def __getattr__(self, key):
    if self.is_struct:
      if key not in self.params:
        return None
      value = self.params[key]
      if type(value) == struct_pb2.Struct:
        return Parameter(value, True, self._l2_reg)
      else:
        return value
    value = getattr(self.params, key)
    if is_proto_message(self.params, key):
      return Parameter(value, False, self._l2_reg)
    return value

  def __getitem__(self, key):
    return self.__getattr__(key)

  def get_or_default(self, key, def_val):
    if self.is_struct:
      if key in self.params:
        if def_val is None:
          return self.params[key]
        value = self.params[key]
        if type(value) == float:
          return type(def_val)(value)
        return value
      return def_val
    else:  # pb message
      value = getattr(self.params, key, def_val)
      if hasattr(value, '__len__'):  # repeated
        return value if len(value) > 0 else def_val
      try:
        if self.params.HasField(key):
          return value
      except ValueError:
        pass
      return def_val  # maybe not equal to the default value of msg field

  def check_required(self, keys):
    if not self.is_struct:
      return
    if not isinstance(keys, (list, tuple)):
      keys = [keys]
    for key in keys:
      if key not in self.params:
        raise KeyError('%s must be set in params' % key)

  def has_field(self, key):
    if self.is_struct:
      return key in self.params
    else:
      return self.params.HasField(key)
