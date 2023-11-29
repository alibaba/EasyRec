# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers.keras.blocks import MLP

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def gate_fn(inputs, units, name, l2_reg, reuse):
  weights = tf.layers.dense(
      inputs,
      units,
      kernel_regularizer=l2_reg,
      name='%s/dense' % name,
      reuse=reuse)
  return tf.nn.softmax(weights, axis=1)


class MMoE(tf.keras.layers.Layer):
  """Multi-gate Mixture-of-Experts model."""

  def __init__(self, params, name='MMoE', reuse=None, **kwargs):
    super(MMoE, self).__init__(name, **kwargs)
    params.check_required(['num_expert', 'num_task'])
    self._reuse = reuse
    self._num_expert = params.num_expert
    self._num_task = params.num_task
    if params.has_field('expert_mlp'):
      expert_params = params.expert_mlp
      self._has_experts = True
      self._experts = [
          MLP(expert_params, 'expert_%d' % i, reuse=reuse)
          for i in range(self._num_expert)
      ]
    else:
      self._has_experts = False
      self._experts = [lambda x: x[i] for i in range(self._num_expert)]
    self._l2_reg = params.l2_regularizer

  def __call__(self, inputs, **kwargs):
    if self._num_expert == 0:
      logging.warning('num_expert of MMoE layer `%s` is 0' % self.name)
      return inputs

    with tf.name_scope(self.name):
      expert_fea_list = [expert(inputs) for expert in self._experts]
      experts_fea = tf.stack(expert_fea_list, axis=1)

      gate_input = inputs if self._has_experts else inputs[self._num_expert]
      task_input_list = []
      for task_id in range(self._num_task):
        gate = gate_fn(
            gate_input,
            self._num_expert,
            name='gate_%d' % task_id,
            l2_reg=self._l2_reg,
            reuse=self._reuse)
        gate = tf.expand_dims(gate, -1)
        task_input = tf.multiply(experts_fea, gate)
        task_input = tf.reduce_sum(task_input, axis=1)
        task_input_list.append(task_input)
    return task_input_list
