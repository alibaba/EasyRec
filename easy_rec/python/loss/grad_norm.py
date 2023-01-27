# -*- encoding: utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


def grad_norm(losses, weights_shared, alpha=0.12):

  def _get_step0_loss(loss, name=None):
    l0 = variable_scope.get_variable(
        name=name,
        shape=[],
        initializer=init_ops.constant_initializer(-1.0),
        trainable=False)

    def _assign_l0():
      with tf.control_dependencies([state_ops.assign(l0, loss)]):
        return array_ops.identity(l0)

    return tf.cond(math_ops.less(l0, 0.0), _assign_l0, lambda: l0)

  N = len(losses)
  logging.info('number of gradnorm losses: %d' % N)
  # L2-norm of gradients of each task loss wrt shared parameters
  Ws = []
  Gs = []
  for tid, l in enumerate(losses):
    Ws.append(
        variable_scope.get_variable(
            name='grad_norm/loss_w_%d' % tid,
            shape=[],
            initializer=init_ops.constant_initializer(1.0),
            dtype=tf.float32))
    losses[tid] = l * Ws[tid]
    G = tf.gradients(losses[tid], weights_shared)
    Gs.append(tf.norm(G, ord=2))

  # Gradient averaged over all tasks
  G_avg = math_ops.add_n(Gs) / N

  # Relative losses L_hat_i(t)
  l_hats = []
  for tid, l in enumerate(losses):
    l_hats.append(l / _get_step0_loss(l, name='grad_norm/loss_%d' % tid))

  l_hat_avg = math_ops.add_n(l_hats) / N

  # Inverse training rates r_i(t)
  loss_gradnorms = []
  for tid, l_hat in enumerate(l_hats):
    inv_rate = l_hat / l_hat_avg
    C = G_avg * tf.pow(inv_rate, alpha)
    C = tf.stop_gradient(C)
    loss_gradnorms.append(tf.abs(Gs[tid] - C))

  loss_gradnorm = math_ops.add_n(loss_gradnorms)

  # Renormalize weights
  with tf.control_dependencies([loss_gradnorm]):
    coef = N * 1.0 / math_ops.add_n(Ws)
    for w in Ws:
      ops.add_to_collection(ops.GraphKeys.UPDATE_OPS,
                            state_ops.assign(w, w * coef))

  return losses, loss_gradnorm
