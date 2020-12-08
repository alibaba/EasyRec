# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MMOE:

  def __init__(self,
               expert_dnn_config,
               l2_reg,
               num_task,
               num_expert=None,
               name='mmoe',
               is_training=False):
    """Initializes a `DNN` Layer.

    Args:
      expert_dnn_config: a instance or a list of easy_rec.python.protos.dnn_pb2.DNN,
        if it is a list of configs, the param `num_expert` will be ignored,
        if it is a single config, the number of experts will be specified by num_expert.
      l2_reg: l2 regularizer.
      num_task: number of tasks
      num_expert: number of experts, default is the list length of expert_dnn_configs
      name: scope of the DNN, so that the parameters could be separated from other dnns
      is_training: train phase or not, impact batchnorm and dropout
    """
    if isinstance(expert_dnn_config, list):
      self._expert_dnn_configs = expert_dnn_config
      self._num_expert = len(expert_dnn_config)
    else:
      assert num_expert is not None and num_expert > 0, \
          'param `num_expert` must be large than zero, when expert_dnn_config is not a list'
      self._expert_dnn_configs = [expert_dnn_config] * num_expert
      self._num_expert = num_expert
    logging.info('num_expert: {0}'.format(self._num_expert))

    self._num_task = num_task
    self._l2_reg = l2_reg
    self._name = name
    self._is_training = is_training

  @property
  def num_expert(self):
    return self._num_expert

  def gate(self, unit, deep_fea, name):
    fea = tf.layers.dense(
        inputs=deep_fea,
        units=unit,
        kernel_regularizer=self._l2_reg,
        name='%s/dnn' % name)
    fea = tf.nn.softmax(fea, axis=1)
    return fea

  def __call__(self, deep_fea):
    expert_fea_list = []
    for expert_id in range(self._num_expert):
      expert_dnn_config = self._expert_dnn_configs[expert_id]
      expert_dnn = dnn.DNN(
          expert_dnn_config,
          self._l2_reg,
          name='%s/expert_%d' % (self._name, expert_id),
          is_training=self._is_training)
      expert_fea = expert_dnn(deep_fea)
      expert_fea_list.append(expert_fea)
    experts_fea = tf.stack(expert_fea_list, axis=1)

    task_input_list = []
    for task_id in range(self._num_task):
      gate = self.gate(
          self._num_expert, deep_fea, name='%s/gate_%d' % (self._name, task_id))
      gate = tf.expand_dims(gate, -1)
      task_input = tf.multiply(experts_fea, gate)
      task_input = tf.reduce_sum(task_input, axis=1)
      task_input_list.append(task_input)
    return task_input_list
