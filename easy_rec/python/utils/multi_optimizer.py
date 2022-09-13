# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Date: 2018-09-13
import tensorflow as tf
from tensorflow.python.training import optimizer


class MultiOptimizer(optimizer.Optimizer):

  def __init__(self, opts, grouped_vars, use_locking=False):
    """Combine multiple optimizers for optimization, such as WideAndDeep.

    Args:
      opts: list of optimizer instance.
      grouped_vars: list of list of vars, each list of vars are
          optimized by each of the optimizers.
      use_locking: be compatible, currently not used.
    """
    super(MultiOptimizer, self).__init__(use_locking, 'MultiOptimizer')
    self._opts = opts
    self._grouped_vars = grouped_vars

  def compute_gradients(self, loss, variables, **kwargs):
    grad_and_vars = []
    for gid, opt in enumerate(self._opts):
      grad_and_vars.extend(
          opt.compute_gradients(loss, self._grouped_vars[gid], **kwargs))
    return grad_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    update_ops = []
    grads_and_vars = [x for x in grads_and_vars]
    for gid, opt in enumerate(self._opts):
      tmp = [x for x in grads_and_vars if x[1] in self._grouped_vars[gid]]
      if gid == 0:
        update_ops.append(opt.apply_gradients(tmp, global_step))
      else:
        update_ops.append(opt.apply_gradients(tmp, None))
    return tf.group(update_ops)

  def open_auto_record(self, flag=True):
    super(MultiOptimizer, self).open_auto_record(flag)

  def get_slot(self, var, name):
    raise NotImplementedError('not implemented')
    # for opt in self._opts:
    #   tmp = opt.get_slot(var, name)
    #   if tmp is not None:
    #     return tmp
    # return None

  def variables(self):
    all_vars = []
    for opt in self._opts:
      all_vars.extend(opt.variables())
    return all_vars

  def get_slot_names(self):
    slot_names = []
    for opt in self._opts:
      slot_names.extend(opt.get_slot_names())
    return slot_names
