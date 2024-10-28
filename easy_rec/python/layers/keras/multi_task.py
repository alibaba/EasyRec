# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras.attention import Attention
from easy_rec.python.layers.keras.blocks import MLP
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.protos import seq_encoder_pb2

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MMoE(Layer):
  """Multi-gate Mixture-of-Experts model."""

  def __init__(self, params, name='MMoE', reuse=None, **kwargs):
    super(MMoE, self).__init__(name=name, **kwargs)
    params.check_required(['num_expert', 'num_task'])
    self._reuse = reuse
    self._num_expert = params.num_expert
    self._num_task = params.num_task
    if params.has_field('expert_mlp'):
      expert_params = Parameter.make_from_pb(params.expert_mlp)
      expert_params.l2_regularizer = params.l2_regularizer
      self._has_experts = True
      self._experts = [
          MLP(expert_params, 'expert_%d' % i, reuse=reuse)
          for i in range(self._num_expert)
      ]
    else:
      self._has_experts = False

    self._gates = []
    for task_id in range(self._num_task):
      dense = Dense(
          self._num_expert,
          activation='softmax',
          name='gate_%d' % task_id,
          kernel_regularizer=params.l2_regularizer)
      self._gates.append(dense)

  def call(self, inputs, training=None, **kwargs):
    if self._num_expert == 0:
      logging.warning('num_expert of MMoE layer `%s` is 0' % self.name)
      return inputs
    if self._has_experts:
      expert_fea_list = [
          expert(inputs, training=training) for expert in self._experts
      ]
    else:
      expert_fea_list = inputs
    experts_fea = tf.stack(expert_fea_list, axis=1)
    # 不使用内置MLP作为expert时，gate的input使用最后一个额外的输入
    gate_input = inputs if self._has_experts else inputs[self._num_expert]
    task_input_list = []
    for task_id in range(self._num_task):
      gate = self._gates[task_id](gate_input)
      gate = tf.expand_dims(gate, -1)
      task_input = tf.multiply(experts_fea, gate)
      task_input = tf.reduce_sum(task_input, axis=1)
      task_input_list.append(task_input)
    return task_input_list


class AITMTower(Layer):
  """Adaptive Information Transfer Multi-task (AITM) Tower."""

  def __init__(self, params, name='AITMTower', reuse=None, **kwargs):
    super(AITMTower, self).__init__(name=name, **kwargs)
    self.project_dim = params.get_or_default('project_dim', None)
    self.stop_gradient = params.get_or_default('stop_gradient', True)
    self.transfer = None
    if params.has_field('transfer_mlp'):
      mlp_cfg = Parameter.make_from_pb(params.transfer_mlp)
      mlp_cfg.l2_regularizer = params.l2_regularizer
      self.transfer = MLP(mlp_cfg, name='transfer')
    self.queries = []
    self.keys = []
    self.values = []
    self.attention = None

  def build(self, input_shape):
    if not isinstance(input_shape, (tuple, list)):
      super(AITMTower, self).build(input_shape)
      return
    dim = self.project_dim if self.project_dim else int(input_shape[0][-1])
    for i in range(len(input_shape)):
      self.queries.append(Dense(dim, name='query_%d' % i))
      self.keys.append(Dense(dim, name='key_%d' % i))
      self.values.append(Dense(dim, name='value_%d' % i))
    attn_cfg = seq_encoder_pb2.Attention()
    attn_cfg.scale_by_dim = True
    attn_params = Parameter.make_from_pb(attn_cfg)
    self.attention = Attention(attn_params)
    super(AITMTower, self).build(input_shape)

  def call(self, inputs, training=None, **kwargs):
    if not isinstance(inputs, (tuple, list)):
      return inputs

    queries = []
    keys = []
    values = []
    for i, tower in enumerate(inputs):
      if i == 0:  # current tower
        queries.append(self.queries[i](tower))
        keys.append(self.keys[i](tower))
        values.append(self.values[i](tower))
      else:
        dep = tf.stop_gradient(tower) if self.stop_gradient else tower
        if self.transfer is not None:
          dep = self.transfer(dep, training=training)
        queries.append(self.queries[i](dep))
        keys.append(self.keys[i](dep))
        values.append(self.values[i](dep))
    query = tf.stack(queries, axis=1)
    key = tf.stack(keys, axis=1)
    value = tf.stack(values, axis=1)
    attn = self.attention([query, value, key])
    return attn[:, 0, :]
