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

import tensorflow as tf
from tensorflow.python.eager import context
# from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
# from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops

from easy_rec.python.compat.dynamic_variable import DynamicVariable


def OptimizerWrapper(optimizer):
  """Abbreviated as ``sok.experiment.OptimizerWrapper``.

  This is a wrapper for tensorflow optimizer so that it can update
  dynamic_variable.DynamicVariable.

  Parameters
  ----------
  optimizer: tensorflow optimizer
      The original tensorflow optimizer.

  Example
  -------
  .. code-block:: python

      import numpy as np
      import tensorflow as tf
      import horovod.tensorflow as hvd
      from sparse_operation_kit import experiment as sok

      v = dynamic_variable.DynamicVariable(dimension=3, initializer="13")

      indices = tf.convert_to_tensor([0, 1, 2**40], dtype=tf.int64)

      with tf.GradientTape() as tape:
          embedding = tf.nn.embedding_lookup(v, indices)
          print("embedding:", embedding)
          loss = tf.reduce_sum(embedding)

      grads = tape.gradient(loss, [v])

      optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
      optimizer = sok.OptimizerWrapper(optimizer)
      optimizer.apply_gradients(zip(grads, [v]))

      embedding = tf.nn.embedding_lookup(v, indices)
      print("embedding:", embedding)
  """
  # a specific code path for dl framework tf2.11.0
  try:
    if isinstance(optimizer, tf.keras.optimizers.legacy.Optimizer):
      return OptimizerWrapperV2(optimizer)
  except Exception:
    pass

  if isinstance(optimizer, tf.keras.optimizers.Optimizer):
    return OptimizerWrapperV2(optimizer)
  else:
    return OptimizerWrapperV1(optimizer)


class OptimizerWrapperV1(object):

  def __init__(self, optimizer):
    self._optimizer = optimizer
    # slots
    unused = tf.Variable([0.0],
                         dtype=tf.float32,
                         name='unused',
                         trainable=False)
    self._optimizer._create_slots([unused])
    names, slots = [], []
    for name in self._optimizer.get_slot_names():
      names.append(name)
      slots.append(self._optimizer.get_slot(unused, name))
    unused_key = self._var_key(unused)
    for name in names:
      assert unused_key in self._optimizer._slots[name]
      self._optimizer._slots[name].pop(unused_key)
    self._initial_vals = {}
    for i, name in enumerate(names):
      self._initial_vals[name] = slots[i]
    # self._optimizer._prepare()

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    self._loss = loss
    tmp_grads = gradients.gradients(loss, var_list)
    return list(zip(tmp_grads, var_list))
    # TODO: the following routine does not work with DynamicVariable
    # return self._optimizer.compute_gradients(loss=loss, var_list=var_list,
    #       # gate_gradients=gate_gradients,
    #       aggregation_method=aggregation_method,
    #       colocate_gradients_with_ops=colocate_gradients_with_ops,
    #       grad_loss=grad_loss)

  def _var_key(self, var):
    if isinstance(var, DynamicVariable):
      return (var._tf_handle.op.graph, var._tf_handle.op.name)
    else:
      return (var.op.graph, var.op.name)

  def _create_slots(self, vars):
    for var in vars:
      if isinstance(var, DynamicVariable):
        self._create_slots_dynamic(var)
      else:
        self._optimizer._create_slots(var)

  def _create_slots_dynamic(self, var):
    key = self._var_key(var)
    for slot_name in self._initial_vals:
      if key not in self._optimizer._slots[slot_name]:
        if var.backend_type == 'hbm':
          with ops.colocate_with(var):
            slot = DynamicVariable(
                dimension=var.dimension,
                initializer=self._initial_vals[slot_name],
                name='DynamicSlot',
                trainable=False)
        else:
          tmp_config = var.config_dict
          # tmp_initializer = var.initializer_str
          with ops.colocate_with(var):
            slot = DynamicVariable(
                dimension=var.dimension,
                initializer=self._initial_vals[slot_name],
                var_type=var.backend_type,
                name='DynamicSlot',
                trainable=False,
                **tmp_config)

        self._optimizer._slots[slot_name][key] = slot

  def get_slot_names(self):
    return self._optimizer.get_slot_names()

  def get_slot(self, var, slot_name):
    key = self._var_key(var)
    return self._optimizer._slots[slot_name][key]

  @property
  def _slots(self):
    return self._optimizer._slots

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    gradients = grads_and_vars
    sparse_vars = [x for x in gradients if 'DynamicVariable' in str(type(x[1]))]
    dense_vars = [
        x for x in gradients if 'DynamicVariable' not in str(type(x[1]))
    ]

    def _dummy_finish(update_ops, name_scope):
      return update_ops

    finish_func = self._optimizer._finish
    self._optimizer._finish = _dummy_finish
    with ops.control_dependencies([array_ops.identity(self._loss)]):
      sparse_grad_updates = self.apply_sparse_gradients(sparse_vars, name=name)

    dense_grad_updates = self._optimizer.apply_gradients(
        dense_vars, global_step=None, name=name)
    if sparse_grad_updates is not None and dense_grad_updates is not None:
      grad_updates = sparse_grad_updates + dense_grad_updates
    elif sparse_grad_updates is not None:
      grad_updates = sparse_grad_updates
    elif dense_grad_updates is not None:
      grad_updates = dense_grad_updates

    assert global_step is not None
    with ops.control_dependencies([finish_func(grad_updates, 'update')]):
      with ops.colocate_with(global_step):
        if isinstance(global_step, resource_variable_ops.BaseResourceVariable):
          # TODO(apassos): the implicit read in assign_add is slow; consider
          # making it less so.
          apply_updates = resource_variable_ops.assign_add_variable_op(
              global_step.handle,
              ops.convert_to_tensor(1, dtype=global_step.dtype),
              name=name)
        else:
          apply_updates = state_ops.assign_add(global_step, 1, name=name)

    if not context.executing_eagerly():
      if isinstance(apply_updates, ops.Tensor):
        apply_updates = apply_updates.op
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      if apply_updates not in train_op:
        train_op.append(apply_updates)

    return apply_updates

  def apply_sparse_gradients(self, grads_and_vars, global_step=None, name=None):
    # 1. Create slots and do sparse_read
    to_static_ops = []
    grad_list, var_list = [], []
    for g, v in grads_and_vars:
      if g is not None:
        unique, indices = tf.unique(g.indices)
        grad_list.append(ops.IndexedSlices(g.values, indices, g.dense_shape))
        # TODO: Check multi-thread safety of DET
        with tf.control_dependencies([g.values]):
          to_static_ops.append(v.to_static(unique, False))
        var_list.append(v)
        key = self._var_key(v)
        for slot_name in self._initial_vals:
          if key not in self._optimizer._slots[slot_name]:
            tmp_slot_var_name = v._dummy_handle.op.name + '/' + self._optimizer._name
            if v.backend_type == 'hbm':
              with ops.colocate_with(v):
                slot = DynamicVariable(
                    dimension=v.dimension,
                    initializer=self._initial_vals[slot_name],
                    name=tmp_slot_var_name,
                    trainable=False,
                )
            else:
              tmp_config = v.config_dict
              # tmp_initializer = v.initializer_str
              with ops.colocate_with(v):
                slot = DynamicVariable(
                    dimension=v.dimension,
                    initializer=self._initial_vals[slot_name],
                    var_type=v.backend_type,
                    name=tmp_slot_var_name,
                    trainable=False,
                    **tmp_config)

            self._optimizer._slots[slot_name][key] = slot
          else:
            slot = self._optimizer._slots[slot_name][key]
          to_static_ops.append(slot.to_static(unique))

    if len(grad_list) == 0:
      return

    # 3. Call tf-optimizer
    with ops.control_dependencies(to_static_ops):
      train_op = self._optimizer.apply_gradients(
          zip(grad_list, var_list), global_step=global_step, name=name)

    # 5. Write buffer back to dynamic variables
    to_dynamic_ops = []
    if not isinstance(train_op, list):
      train_op = [train_op]
    with ops.control_dependencies(train_op):
      for v in var_list:
        key = self._var_key(v)
        to_dynamic_ops.append(v.to_dynamic())
        for name in self._initial_vals:
          slot = self._optimizer._slots[name][key]
          to_dynamic_ops.append(slot.to_dynamic())

    return to_dynamic_ops


class OptimizerWrapperV2(object):

  def __init__(self, optimizer):
    self._optimizer = optimizer
    # slots
    if tf.__version__[0] == '1':
      unused = tf.Variable([0.0],
                           name='unused',
                           trainable=False,
                           use_resource=True)
    else:
      unused = tf.Variable([0.0], name='unused', trainable=False)
    self._optimizer._create_slots([unused])
    names, slots = [], []
    for name in self._optimizer.get_slot_names():
      names.append(name)
      slots.append(self._optimizer.get_slot(unused, name))
    unused_key = self._var_key(unused)
    if unused_key in self._optimizer._slots:
      self._optimizer._slots.pop(unused_key)
    self._initial_vals = {}
    for i, name in enumerate(names):
      self._initial_vals[name] = slots[i]
    self._iterations = tf.Variable(0)

  @property
  def lr(self):
    return self._optimizer.lr

  def _create_slots(self, vars):
    for tmp_var in vars:
      if isinstance(tmp_var, DynamicVariable):
        self._create_slots_dynamic(tmp_var)
      else:
        self._optimizer._create_slots(tmp_var)

  def _create_slots_dynamic(self, var):
    key = self._var_key(var)
    if key not in self._optimizer._slots:
      self._optimizer._slots[key] = {}
    for slot_name in self._initial_vals:
      if slot_name not in self._optimizer._slots[key]:
        if var.backend_type == 'hbm':
          slot = DynamicVariable(
              dimension=var.dimension,
              initializer=self._initial_vals[slot_name],
              name='DynamicSlot',
              trainable=False,
          )
        else:
          tmp_config = var.config_dict
          # tmp_initializer = var.initializer_str
          slot = DynamicVariable(
              dimension=var.dimension,
              initializer=self._initial_vals[slot_name],
              var_type=var.backend_type,
              name='DynamicSlot',
              trainable=False,
              **tmp_config)
        self._optimizer._slots[key][slot_name] = slot

  def _var_key(self, var):
    if hasattr(var, '_distributed_container'):
      var = var._distributed_container()
    if var._in_graph_mode:
      return var._shared_name
    return var._unique_id

  def get_slot_names(self):
    return self._optimizer.get_slot_names()

  def get_slot(self, var, name):
    return self._optimizer.get_slot(var, name)

  @property
  def _slots(self):
    return self._optimizer._slots

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # 1. Create slots and do sparse_read
    to_static_ops = []
    grad_list, var_list = [], []
    for g, v in grads_and_vars:
      if g is not None:
        unique, indices = tf.unique(g.indices)
        grad_list.append(ops.IndexedSlices(g.values, indices, g.dense_shape))
        # TODO: Check multi-thread safety of DET
        # with tf.control_dependencies([g.values]):
        to_static_ops.append(v.to_static(unique))
        var_list.append(v)
        key = self._var_key(v)
        if key not in self._optimizer._slots:
          self._optimizer._slots[key] = {}
        for slot_name in self._initial_vals:
          if slot_name not in self._optimizer._slots[key]:
            if v.backend_type == 'hbm':
              slot = DynamicVariable(
                  dimension=v.dimension,
                  initializer=self._initial_vals[slot_name],
                  name='DynamicSlot',
                  trainable=False,
              )
            else:
              tmp_config = v.config_dict
              # tmp_initializer = v.initializer_str
              slot = DynamicVariable(
                  dimension=v.dimension,
                  initializer=self._initial_vals[slot_name],
                  var_type=v.backend_type,
                  name='DynamicSlot',
                  trainable=False,
                  **tmp_config)

            self._optimizer._slots[key][slot_name] = slot
          else:
            slot = self._optimizer._slots[key][slot_name]
          to_static_ops.append(slot.to_static(unique))

    if len(grad_list) == 0:
      return

    # 2. Switch iterations
    iterations = self._optimizer._iterations
    self._optimizer._iterations = self._iterations

    # 3. Call tf-optimizer
    with tf.control_dependencies(to_static_ops):
      train_op = self._optimizer.apply_gradients(
          zip(grad_list, var_list), name=name)

    # 4. Switch iterations
    self._optimizer._iterations = iterations

    # 5. Write buffer back to dynamic variables
    to_dynamic_ops = []
    with tf.control_dependencies([train_op]):
      for v in var_list:
        key = self._var_key(v)
        to_dynamic_ops.append(v.to_dynamic())
        for name in self._initial_vals:
          slot = self._optimizer._slots[key][name]
          to_dynamic_ops.append(slot.to_dynamic())
    return tf.group(to_dynamic_ops)


class SGD(object):

  def __init__(self, lr):
    self._lr = tf.Variable(lr)

  @property
  def lr(self):
    return self._lr

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    train_ops = []
    for g, v in grads_and_vars:
      if g is not None:
        scaled_g = ops.IndexedSlices(g.values * self._lr, g.indices,
                                     g.dense_shape)
        train_ops.append(v.scatter_sub(scaled_g))
    return tf.group(train_ops)
