# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import mmoe
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.protos.mmoe_pb2 import MMoE as MMoEConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MMoE(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MMoE, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert self._model_config.WhichOneof('model') == 'mmoe', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.mmoe
    assert isinstance(self._model_config, MMoEConfig)

    if self.has_backbone:
      self._features = self.backbone
    else:
      self._features, _ = self._input_layer(self._feature_dict, 'all')
    self._init_towers(self._model_config.task_towers)

  def build_predict_graph(self):
    if self._model_config.HasField('expert_dnn'):
      mmoe_layer = mmoe.MMOE(
          self._model_config.expert_dnn,
          l2_reg=self._l2_reg,
          num_task=self._task_num,
          num_expert=self._model_config.num_expert)
    else:
      # For backward compatibility with original mmoe layer config
      mmoe_layer = mmoe.MMOE([x.dnn for x in self._model_config.experts],
                             l2_reg=self._l2_reg,
                             num_task=self._task_num)
    task_input_list = mmoe_layer(self._features)

    tower_outputs = {}
    for i, task_tower_cfg in enumerate(self._model_config.task_towers):
      tower_name = task_tower_cfg.tower_name

      if task_tower_cfg.HasField('dnn'):
        tower_dnn = dnn.DNN(
            task_tower_cfg.dnn,
            self._l2_reg,
            name=tower_name,
            is_training=self._is_training)
        tower_output = tower_dnn(task_input_list[i])
      else:
        tower_output = task_input_list[i]
      tower_output = tf.layers.dense(
          inputs=tower_output,
          units=task_tower_cfg.num_class,
          kernel_regularizer=self._l2_reg,
          name='dnn_output_%d' % i)

      tower_outputs[tower_name] = tower_output
    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict
