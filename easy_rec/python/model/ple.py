# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.protos.ple_pb2 import PLE as PLEConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class PLE(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(PLE, self).__init__(model_config, feature_configs, features, labels,
                              is_training)
    assert self._model_config.WhichOneof('model') == 'ple', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.ple
    assert isinstance(self._model_config, PLEConfig)

    self._layer_nums = len(self._model_config.extraction_networks)
    self._task_nums = len(self._model_config.task_towers)
    if self.has_backbone:
      self._features = self.backbone
    else:
      self._features, _ = self._input_layer(self._feature_dict, 'all')
    self._init_towers(self._model_config.task_towers)

  def gate(self, selector_fea, vec_feas, name):
    vec = tf.stack(vec_feas, axis=1)
    gate = tf.layers.dense(
        inputs=selector_fea,
        units=len(vec_feas),
        kernel_regularizer=self._l2_reg,
        activation=None,
        name=name + '_gate/dnn')
    gate = tf.nn.softmax(gate, axis=1)
    gate = tf.expand_dims(gate, -1)
    task_input = tf.multiply(vec, gate)
    task_input = tf.reduce_sum(task_input, axis=1)
    return task_input

  def experts_layer(self, deep_fea, expert_num, experts_cfg, name):
    tower_outputs = []
    for expert_id in range(expert_num):
      tower_dnn = dnn.DNN(
          experts_cfg,
          self._l2_reg,
          name=name + '_expert_%d/dnn' % expert_id,
          is_training=self._is_training)
      tower_output = tower_dnn(deep_fea)
      tower_outputs.append(tower_output)
    return tower_outputs

  def CGC_layer(self, extraction_networks_cfg, extraction_network_fea,
                shared_expert_fea, final_flag):
    layer_name = extraction_networks_cfg.network_name
    expert_shared_out = self.experts_layer(
        shared_expert_fea, extraction_networks_cfg.share_num,
        extraction_networks_cfg.share_expert_net, layer_name + '_share/dnn')

    experts_outs = []
    cgc_layer_outs = []
    for task_idx in range(self._task_nums):
      name = layer_name + '_task_%d' % task_idx
      experts_out = self.experts_layer(
          extraction_network_fea[task_idx],
          extraction_networks_cfg.expert_num_per_task,
          extraction_networks_cfg.task_expert_net, name)
      cgc_layer_out = self.gate(extraction_network_fea[task_idx],
                                experts_out + expert_shared_out, name)
      experts_outs.extend(experts_out)
      cgc_layer_outs.append(cgc_layer_out)

    if final_flag:
      shared_layer_out = None
    else:
      shared_layer_out = self.gate(shared_expert_fea,
                                   experts_outs + expert_shared_out,
                                   layer_name + '_share')
    return cgc_layer_outs, shared_layer_out

  def build_predict_graph(self):
    extraction_network_fea = [self._features] * self._task_nums
    shared_expert_fea = self._features
    final_flag = False
    for idx in range(len(self._model_config.extraction_networks)):
      extraction_network = self._model_config.extraction_networks[idx]
      if idx == len(self._model_config.extraction_networks) - 1:
        final_flag = True
      extraction_network_fea, shared_expert_fea = self.CGC_layer(
          extraction_network, extraction_network_fea, shared_expert_fea,
          final_flag)

    tower_outputs = {}
    for i, task_tower_cfg in enumerate(self._model_config.task_towers):
      tower_name = task_tower_cfg.tower_name
      tower_dnn = dnn.DNN(
          task_tower_cfg.dnn,
          self._l2_reg,
          name=tower_name,
          is_training=self._is_training)
      tower_output = tower_dnn(extraction_network_fea[i])

      tower_output = tf.layers.dense(
          inputs=tower_output,
          units=task_tower_cfg.num_class,
          kernel_regularizer=self._l2_reg,
          name='dnn_output_%d' % i)

      tower_outputs[tower_name] = tower_output
    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict
