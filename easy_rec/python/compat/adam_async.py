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
"""AdamAsync for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework.load_library import load_op_library
# from tensorflow.python.ops import kv_variable_ops
# from tensorflow.python.ops import gen_hash_training_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator
from tensorflow.python.training import training_ops
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import tf_export

curr_dir, _ = os.path.split(__file__)
adam_async_ops = load_op_library(os.path.join(curr_dir, 'libadam_async_op.so'))


@tf_export('train.AdamAsyncOptimizer')
class AdamAsyncOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adam algorithm.

  See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               use_locking=False,
               apply_sparse_rmsprop=False,
               name='Adam'):
    """Construct a new Adam optimizer for training asynchronous.

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    v_0 <- 0 (Initialize initial 2nd moment vector)
    t <- 0 (Initialize timestep)
    ```

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    ```
    t <- t + 1
    lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
    variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
    ```

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      apply_sparse_rmsprop: If True use rmsprop optimizer for sparse apply.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AdamASync".
    """
    super(AdamAsyncOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Created in SparseApply if needed.
    self._updated_lr = None

    self._apply_sparse_rmsprop = apply_sparse_rmsprop

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable('beta1_power', graph=graph),
              self._get_non_slot_variable('beta2_power', graph=graph))

  def _create_slots(self, var_list):
    # When training asynchronous, we create the beta1 and beta2 accumulators for
    # each variable to avoid communication bottlenecks.
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name='beta1_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name='beta2_power', colocate_with=first_var)

    # Create slots for the moments.
    for v in var_list:
      with ops.colocate_with(v):
        self._zeros_slot(v, 'm', self._name)
        self._zeros_slot(v, 'v', self._name)
        # self._get_or_make_slot(v,
        #     ops.convert_to_tensor(self._beta1, dtype=v.dtype.base_dtype),
        #     "beta1_power", self._name)
        # self._get_or_make_slot(v,
        #     ops.convert_to_tensor(self._beta2, dtype=v.dtype.base_dtype),
        #     "beta2_power", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name='learning_rate')
    self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1')
    self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2')
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name='epsilon')

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    # beta1_power = self.get_slot(var, 'beta1_power')
    # beta2_power = self.get_slot(var, 'beta2_power')
    beta1_power, beta2_power = self._get_beta_accumulators()
    return adam_async_ops.apply_adam_async(
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    # beta1_power = self.get_slot(var, 'beta1_power')
    # beta2_power = self.get_slot(var, 'beta2_power')
    beta1_power, beta2_power = self._get_beta_accumulators()
    return adam_async_ops.resource_apply_adam_async(
        var.handle,
        m.handle,
        v.handle,
        beta1_power.handle,
        beta2_power.handle,
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    # beta1_power = self.get_slot(var, 'beta1_power')
    # beta2_power = self.get_slot(var, 'beta2_power')
    beta1_power, beta2_power = self._get_beta_accumulators()
    return adam_async_ops.sparse_apply_adam_async(
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking,
        apply_sparse_rmsprop=self._apply_sparse_rmsprop)

  # def _hash_table_apply_sparse(self, grad, var, indices):
  #   m = self.get_slot(var, "m")
  #   v = self.get_slot(var, "v")
  #   beta1_power = self.get_slot(var, "beta1_power")
  #   beta2_power = self.get_slot(var, "beta2_power")
  #   update_op = gen_hash_training_ops.tensible_variable_apply_adam(
  #       var.handle, m.handle, v.handle,
  #       math_ops.cast(beta1_power, grad.dtype.base_dtype),
  #       math_ops.cast(beta2_power, grad.dtype.base_dtype),
  #       math_ops.cast(self._lr_t, grad.dtype.base_dtype),
  #       math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
  #       math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
  #       math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
  #       grad, indices, use_locking=self._use_locking)
  #   with ops.control_dependencies([update_op]):
  #     update_beta1 = beta1_power.assign(
  #         beta1_power * math_ops.cast(self._beta1_t, var.dtype.base_dtype),
  #         use_locking=self._use_locking)
  #     update_beta2 = beta2_power.assign(
  #         beta2_power * math_ops.cast(self._beta2_t, var.dtype.base_dtype),
  #         use_locking=self._use_locking)
  #   return control_flow_ops.group(update_beta1, update_beta2)

  def _resource_apply_sparse(self, grad, var, indices):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    # beta1_power = self.get_slot(var, 'beta1_power')
    # beta2_power = self.get_slot(var, 'beta2_power')
    beta1_power, beta2_power = self._get_beta_accumulators()
    return adam_async_ops.resource_sparse_apply_adam_async(
        var.handle,
        m.handle,
        v.handle,
        beta1_power.handle,
        beta2_power.handle,
        math_ops.cast(self._lr_t, grad.dtype),
        math_ops.cast(self._beta1_t, grad.dtype),
        math_ops.cast(self._beta2_t, grad.dtype),
        math_ops.cast(self._epsilon_t, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking,
        apply_sparse_rmsprop=self._apply_sparse_rmsprop)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)
