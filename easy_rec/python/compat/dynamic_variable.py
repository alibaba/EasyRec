#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
#

import json

import tensorflow as tf
from sparse_operation_kit.experiment import raw_ops as dynamic_variable_ops
from sparse_operation_kit.experiment.communication import num_gpus
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.ops.resource_variable_ops import variable_accessed

# from tensorflow.python.util import object_identity

dynamic_variable_count = 0

_resource_var_from_proto = ResourceVariable.from_proto


class DynamicVariable(ResourceVariable):
  """Abbreviated as ``sok.experiment.DynamicVariable``.

  A variable that allocates memory dynamically.

  Parameters
  ----------
  dimension: int
      The last dimension of this variable(that is, the embedding vector
      size of embedding table).

  initializer: string
      a string to specify how to initialize this variable.
      Currently, only support "random" or string of a float
      value(meaning const initializer). Default value is "random".

  var_type: string
      a string to specify to use DET or HKV as the backend.
      If use HKV as the backend, only support tf.int64 as key_type
      If use HKV as the backend, please set init_capacity and max_capacity value equal to 2 powers.

  key_type: dtype
      specify the data type of indices. Unlike the static variable of
      tensorflow, this variable is dynamically allocated and contains
      a hash table inside it. So the data type of indices must be
      specified to construct the hash table. Default value is tf.int64.

  dtype: dtype
      specify the data type of values. Default value is tf.float32.

  Example
  -------
  .. code-block:: python

      import numpy as np
      import tensorflow as tf
      import horovod.tensorflow as hvd
      from sparse_operation_kit import experiment as sok

      v = sok.DynamicVariable(dimension=3, initializer="13")
      print("v.shape:", v.shape)
      print("v.size:", v.size)

      indices = tf.convert_to_tensor([0, 1, 2**40], dtype=tf.int64)

      embedding = tf.nn.embedding_lookup(v, indices)
      print("embedding:", embedding)
      print("v.shape:", v.shape)
      print("v.size:", v.size)
  """

  def __init__(self,
               dimension,
               initializer=None,
               var_type=None,
               name=None,
               constraint=None,
               trainable=True,
               key_type=None,
               dtype=None,
               mode=None,
               variable_def=None,
               import_scope=None,
               **kwargs):
    self._indices = None
    if variable_def is not None:
      super(DynamicVariable, self)._init_from_proto(
          variable_def, import_scope=import_scope, validate_shape=False)
      g = ops.get_default_graph()
      handle = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.variable_name, import_scope=import_scope),
          allow_operation=False)
      self._dimension = handle.op.get_attr('shape').dim[-1].size
      self._key_type = handle.op.get_attr('key_type')
      self._handle_type = handle.op.get_attr('dtype')
      self._mode = None
      self._config = {}
      self._name = variable_def.variable_name.split(':')[0]
      self._trainable = variable_def.trainable
      self._dummy_handle = handle
      self._handle = handle

      # init op
      init_op = g.as_graph_element(variable_def.initializer_name)
      self._initializer_op = init_op

      init_tf = init_op.control_inputs[0]
      # init_dummy = init_op.control_inputs[1]

      self._tf_handle = init_tf.inputs[0]
      return

    self._key_type = key_type if key_type is not None else tf.int64
    self._handle_dtype = dtype if dtype is not None else tf.float32
    self._dimension = dimension
    self._mode = mode
    self._config = json.dumps(kwargs)
    self._config_dict = kwargs
    if var_type == 'hybrid' and self._key_type != tf.int64:
      raise NotImplementedError(
          'only key_type tf.int64 is supported in HKV backend')
    if name is None:
      global dynamic_variable_count
      name = 'sok_dynamic_Variable_' + str(dynamic_variable_count)
      dynamic_variable_count += 1
    var_type = 'hbm' if var_type is None else var_type
    self._var_type = var_type
    self._base = super(DynamicVariable, self)
    self._base.__init__(
        initial_value=[[0.0] * dimension],
        trainable=trainable,
        name=name + '/proxy',
        dtype=self._handle_dtype,
        constraint=constraint,
        distribute_strategy=None,
        synchronization=None,
        aggregation=None,
        shape=[None, dimension],
    )

    with ops.init_scope():
      # name = "DynamicVariable" if name is None else name
      with ops.name_scope(name) as name_scope:
        self._dummy_name = ops.name_from_scope_name(name_scope)
        if context.executing_eagerly():
          self._dummy_name = '%s_%d' % (name, ops.uid())
        with ops.NullContextmanager():
          shape = [None, dimension]
          initializer = '' if initializer is None else initializer
          self._initializer = initializer
          handle = dynamic_variable_ops.dummy_var_handle(
              container='DummyVariableContainer',
              shared_name=self._dummy_name,
              key_type=self._key_type,
              dtype=self._handle_dtype,
              shape=shape,
          )
          if type(initializer) is str:
            init_op = dynamic_variable_ops.dummy_var_initialize(
                handle,
                initializer=initializer,
                var_type=var_type,
                unique_name=self._dummy_name,
                key_type=self._key_type,
                dtype=self._handle_dtype,
                config=self._config,
            )
          else:
            with tf.control_dependencies([initializer._initializer_op]):
              initial_val = initializer.read_value()
            init_op = dynamic_variable_ops.dummy_var_initialize(
                handle,
                initializer=initial_val,
                var_type=var_type,
                unique_name=self._dummy_name,
                key_type=self._key_type,
                dtype=self._handle_dtype,
                config=self._config,
            )
          # TODO: Add is_initialized_op
          # is_initialized_op = ops.convert_to_tensor(True)

          self._tf_handle = self._handle
          self._dummy_handle = handle
          # Note that the default handle will be sok's handle
          self._handle = self._dummy_handle
          self._initializer_op = tf.group([self._initializer_op, init_op])
          # self._is_initialized_op = tf.group([self._is_initialized_op, is_initialized_op])

      handle_data = (
          resource_variable_ops.cpp_shape_inference_pb2.CppShapeInferenceResult
          .HandleData())
      handle_data.is_set = True
      handle_data.shape_and_type.append(
          resource_variable_ops.cpp_shape_inference_pb2.CppShapeInferenceResult
          .HandleShapeAndType(
              shape=self.shape.as_proto(), dtype=self.dtype.as_datatype_enum))
      resource_variable_ops._set_handle_shapes_and_types(
          self._handle,
          handle_data,
          graph_mode=False if context.executing_eagerly() else True)

  def is_static(self):
    return self._handle is self._tf_handle

  def to_static(self, indices, lookup_only=False):
    if not self.is_static() and self._indices is None:
      buffer = self.sparse_read(indices, lookup_only)
      self._indices = indices
      self._handle = self._tf_handle
      return self.assign(buffer)
    else:
      raise RuntimeError('to_static() must be called in dynamic mode.')

  def to_dynamic(self):
    if self.is_static():
      buffer = self.read_value()
      sparse_delta = ops.IndexedSlices(buffer, self._indices, self.shape)
      self._indices = None
      self._handle = self._dummy_handle
      return self.scatter_update(sparse_delta)
    else:
      raise RuntimeError('to_dynamic() must be called in static mode.')

  @property
  def name(self):
    return self._dummy_handle.name

  def __repr__(self):
    if self.is_static():
      return self._base.__repr__()
    return "<sok.DynamicVariable '%s' shape=%s dtype=%s>" % (
        self._dummy_name,
        self.shape,
        self.dtype.name,
    )

  @property
  def size(self):
    return dynamic_variable_ops.dummy_var_shape(
        self._dummy_handle, key_type=self._key_type, dtype=self._handle_dtype)

  @property
  def indices(self):
    return self._indices

  @property
  def dimension(self):
    return self._dimension

  def get_shape(self):
    return [self._dimension]

  @property
  def key_type(self):
    return self._key_type

  @property
  def handle_dtype(self):
    return self._handle_dtype

  @property
  def backend_type(self):
    return self._var_type

  @property
  def config_dict(self):
    return self._config_dict

  @property
  def mode(self):
    return self._mode

  @property
  def num_gpus(self):
    return num_gpus()

  @property
  def initializer_str(self):
    return self._initializer

  def key_map(self, indices):
    return indices

  # -------------------------------------------------------------------------
  # Methods supported both in static mode and dynamic mode
  # -------------------------------------------------------------------------

  def sparse_read(self, indices, name=None, lookup_only=False):
    if self.is_static():
      return self._base.sparse_read(indices, name)

    variable_accessed(self)
    if indices.dtype == tf.int32:
      indices = tf.cast(indices, tf.int64)
    return dynamic_variable_ops.dummy_var_sparse_read(
        self._dummy_handle,
        indices,
        dtype=self._handle_dtype,
        lookup_only=lookup_only)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    if self.is_static():
      return self._base.scatter_sub(sparse_delta, use_locking, name)
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
    return dynamic_variable_ops.dummy_var_scatter_add(
        self._dummy_handle,
        sparse_delta.indices,
        ops.convert_to_tensor(-sparse_delta.values, self.dtype),
    )

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if self.is_static():
      return self._base.scatter_add(sparse_delta, use_locking, name)
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
    return dynamic_variable_ops.dummy_var_scatter_add(
        self._dummy_handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
    )

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if self.is_static():
      return self._base.scatter_update(sparse_delta, use_locking, name)
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
    return dynamic_variable_ops.dummy_var_scatter_update(
        self._dummy_handle,
        sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype),
    )

  # -------------------------------------------------------------------------
  # Methods not supported both in static mode and dynamic mode
  # -------------------------------------------------------------------------

  def __deepcopy__(self, *args, **kwargs):
    raise NotImplementedError('__deepcopy__() is not supported.')

  def __reduce__(self, *args, **kwargs):
    raise NotImplementedError('__reduce__() is not supported.')

  def to_proto(self, *args, **kwargs):
    return super(DynamicVariable, self).to_proto(*args, **kwargs)
    # raise NotImplementedError("to_proto() is not supported.")

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if '/DummyVarHandle' in variable_def.variable_name:
      return DynamicVariable(
          dimension=0, variable_def=variable_def, import_scope=import_scope)
    else:
      return _resource_var_from_proto(variable_def, import_scope)
    # raise NotImplementedError("from_proto() is not supported.")

  def set_shape(self, *args, **kwargs):
    raise NotImplementedError('set_shape() is not supported.')

  # -------------------------------------------------------------------------
  # Methods only supported in static mode
  # -------------------------------------------------------------------------

  def is_initialized(self, name):
    return True
    if self.is_static():
      return self._base.is_initialized(name)
    raise NotImplementedError(
        'is_initialized() is not supported in dynamic mode.')

  def _read_variable_op(self):
    if self.is_static():
      return self._base._read_variable_op()
    raise NotImplementedError(
        '_read_variable_op() is not supported in dynamic mode.')

  def value(self):
    if self.is_static():
      return self._base.value()
    raise NotImplementedError('value() is not supported in dynamic mode.')

  def _dense_var_to_tensor(self, *args, **kwargs):
    if self.is_static():
      return self._base._dense_var_to_tensor(*args, **kwargs)
    raise NotImplementedError(
        '_dense_var_to_tensor() is not supported in dynamic mode.')

  def _gather_saveables_for_checkpoint(self):
    if self.is_static():
      return self._base._gather_saveables_for_checkpoint()
    raise NotImplementedError(
        '_gather_saveables_for_checkpoint() is not supported in dynamic mode.')

  def gather_nd(self, *args, **kwargs):
    if self.is_static():
      return self._base.gather_nd(*args, **kwargs)
    raise NotImplementedError('gather_nd() is not supported in dynamic mode.')

  def assign_add(self, *args, **kwargs):
    if self.is_static():
      return self._base.assign_add(*args, **kwargs)
    raise NotImplementedError('assign_add() is not supported in dynamic mode.')

  def assign(self, *args, **kwargs):
    if self.is_static():
      return self._base.assign(*args, **kwargs)
    raise NotImplementedError('assign() is not supported in dynamic mode.')

  def scatter_max(self, *args, **kwargs):
    if self.is_static():
      return self._base.scatter_max(*args, **kwargs)
    raise NotImplementedError('scatter_max() is not supported in dynamic mode.')

  def scatter_min(self, *args, **kwargs):
    if self.is_static():
      return self._base.scatter_min(*args, **kwargs)
    raise NotImplementedError('scatter_min() is not supported in dynamic mode.')

  def scatter_mul(self, *args, **kwargs):
    if self.is_static():
      return self._base.scatter_mul(*args, **kwargs)
    raise NotImplementedError('scatter_mul() is not supported in dynamic mode.')

  def scatter_dim(self, *args, **kwargs):
    if self.is_static():
      return self._base.scatter_dim(*args, **kwargs)
    raise NotImplementedError('scatter_dim() is not supported in dynamic mode.')

  def batch_scatter_update(self, *args, **kwargs):
    if self.is_static():
      return self._base.batch_scatter_update(*args, **kwargs)
    raise NotImplementedError(
        'batch_scatter_update() is not supported in dynamic mode.')

  def scatter_nd_sub(self, *args, **kwargs):
    if self.is_static():
      return self._base.scatter_nd_sub(*args, **kwargs)
    raise NotImplementedError(
        'scatter_nd_sub() is not supported in dynamic mode.')

  def scatter_nd_update(self, *args, **kwargs):
    if self.is_static():
      return self._base.scatter_nd_update(*args, **kwargs)
    raise NotImplementedError(
        'scatter_nd_update() is not supported in dynamic mode.')

  def _strided_slice_assign(self, *args, **kwargs):
    if self.is_static():
      return self._base._strided_slice_assign(*args, **kwargs)
    raise NotImplementedError(
        '_strided_slice_assign() is not supported in dynamic mode.')

  def __int__(self, *args, **kwargs):
    if self.is_static():
      return self._base.__int__(*args, **kwargs)
    raise NotImplementedError('__int__() is not supported in dynamic mode.')


ResourceVariable.from_proto = DynamicVariable.from_proto

# @tf.RegisterGradient("DummyVarSparseRead")
# def _SparseReadGrad(op, grad):
#     """Gradient for sparse_read."""
#     handle = op.inputs[0]
#     indices = op.inputs[1]
#     key_type = op.get_attr("key_type")
#     dtype = op.get_attr("dtype")
#     variable_shape = dynamic_variable_ops.dummy_var_shape(handle, key_type=key_type, dtype=dtype)
#     size = array_ops.expand_dims(array_ops.size(indices), 0)
#     values_shape = array_ops.concat([size, variable_shape[1:]], 0)
#     grad = array_ops.reshape(grad, values_shape)
#     indices = array_ops.reshape(indices, size)
#     return (ops.IndexedSlices(grad, indices, variable_shape), None)


def export(var):
  """Abbreviated as ``sok.experiment.export``.

  Export the indices and value tensor from the given variable.

  Parameters
  ----------
  var: sok.DynamicVariable
      The variable to extract indices and values.

  Returns
  -------
  indices: tf.Tensor
      The indices of the given variable.

  values: tf.Tensor
      the values of the given variable.
  """
  if isinstance(var, DynamicVariable):
    indices, values = dynamic_variable_ops.dummy_var_export(
        var.handle, key_type=var.key_type, dtype=var.handle_dtype)
    with tf.device('CPU'):
      indices = tf.identity(indices)
      values = tf.identity(values)
    return indices, values


def assign(var, indices, values):
  """Abbreviated as ``sok.experiment.assign``.

  Assign the indices and value tensor to the target variable.

  Parameters
  ----------
  var: sok.DynamicVariable
      The target variable of assign.

  indices: tf.Tensor
      indices to be assigned to the variable.

  values: tf.Tensor
      values to be assigned to the variable

  Returns
  -------
  variable: sok.DynamicVariable
  """
  if isinstance(var, DynamicVariable):
    tf.cast(indices, var._key_type)
    return dynamic_variable_ops.dummy_var_assign(var.handle, indices, values)
