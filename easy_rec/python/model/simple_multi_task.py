# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.model.multi_task_model import MultiTaskModel

from easy_rec.python.protos.simple_multi_task_pb2 import SimpleMultiTask as SimpleMultiTaskConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SimpleMultiTask(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(SimpleMultiTask, self).__init__(model_config, feature_configs,
                                          features, labels, is_training)

    assert self._model_config.WhichOneof('model') == 'simple_multi_task', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.simple_multi_task
    assert isinstance(self._model_config, SimpleMultiTaskConfig)

    self._features, _ = self._input_layer(self._feature_dict, 'all')
    regularizers.apply_regularization(
        self._emb_reg, weights_list=[self._features])

    self._init_towers(self._model_config.task_towers)
    self._l2_reg = regularizers.l2_regularizer(
        self._model_config.l2_regularization)

  def build_predict_graph(self):
    tower_outputs = {}
    for i, task_tower_cfg in enumerate(self._task_towers):
      tower_name = task_tower_cfg.tower_name
      task_dnn = dnn.DNN(
          task_tower_cfg.dnn,
          self._l2_reg,
          name=tower_name,
          is_training=self._is_training)
      task_fea = task_dnn(self._features)
      task_output = tf.layers.dense(
          inputs=task_fea,
          units=self._num_class,
          kernel_regularizer=self._l2_reg,
          name='dnn_output_%d' % i)
      tower_outputs[tower_name] = task_output

    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict
