# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import cmbf
from easy_rec.python.layers import dnn
from easy_rec.python.layers import mmoe
from easy_rec.python.layers import uniter
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.protos.dbmtl_pb2 import DBMTL as DBMTLConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DBMTL(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DBMTL, self).__init__(model_config, feature_configs, features, labels,
                                is_training)
    assert self._model_config.WhichOneof('model') == 'dbmtl', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dbmtl
    assert isinstance(self._model_config, DBMTLConfig)

    if self._model_config.HasField('bottom_cmbf'):
      self._cmbf_layer = cmbf.CMBF(model_config, feature_configs, features,
                                   self._model_config.bottom_cmbf,
                                   self._input_layer)
    elif self._model_config.HasField('bottom_uniter'):
      self._uniter_layer = uniter.Uniter(model_config, feature_configs,
                                         features,
                                         self._model_config.bottom_uniter,
                                         self._input_layer)
    elif not self.has_backbone:
      self._features, self._feature_list = self._input_layer(
          self._feature_dict, 'all')
    else:
      assert False, 'invalid code branch'
    self._init_towers(self._model_config.task_towers)

  def build_predict_graph(self):
    bottom_fea = self.backbone
    if bottom_fea is None:
      if self._model_config.HasField('bottom_cmbf'):
        bottom_fea = self._cmbf_layer(self._is_training, l2_reg=self._l2_reg)
      elif self._model_config.HasField('bottom_uniter'):
        bottom_fea = self._uniter_layer(self._is_training, l2_reg=self._l2_reg)
      elif self._model_config.HasField('bottom_dnn'):
        bottom_dnn = dnn.DNN(
            self._model_config.bottom_dnn,
            self._l2_reg,
            name='bottom_dnn',
            is_training=self._is_training)
        bottom_fea = bottom_dnn(self._features)
      else:
        bottom_fea = self._features

    # MMOE block
    if self._model_config.HasField('expert_dnn'):
      mmoe_layer = mmoe.MMOE(
          self._model_config.expert_dnn,
          l2_reg=self._l2_reg,
          num_task=self._task_num,
          num_expert=self._model_config.num_expert)
      task_input_list = mmoe_layer(bottom_fea)
    else:
      task_input_list = [bottom_fea] * self._task_num

    tower_features = {}
    # task specify network
    for i, task_tower_cfg in enumerate(self._model_config.task_towers):
      tower_name = task_tower_cfg.tower_name
      if task_tower_cfg.HasField('dnn'):
        tower_dnn = dnn.DNN(
            task_tower_cfg.dnn,
            self._l2_reg,
            name=tower_name + '/dnn',
            is_training=self._is_training)
        tower_fea = tower_dnn(task_input_list[i])
        tower_features[tower_name] = tower_fea
      else:
        tower_features[tower_name] = task_input_list[i]

    tower_outputs = {}
    relation_features = {}
    # bayes network
    for task_tower_cfg in self._model_config.task_towers:
      tower_name = task_tower_cfg.tower_name
      relation_dnn = dnn.DNN(
          task_tower_cfg.relation_dnn,
          self._l2_reg,
          name=tower_name + '/relation_dnn',
          is_training=self._is_training)
      tower_inputs = [tower_features[tower_name]]
      for relation_tower_name in task_tower_cfg.relation_tower_names:
        tower_inputs.append(relation_features[relation_tower_name])
      relation_input = tf.concat(
          tower_inputs, axis=-1, name=tower_name + '/relation_input')
      relation_fea = relation_dnn(relation_input)
      relation_features[tower_name] = relation_fea

      output_logits = tf.layers.dense(
          relation_fea,
          task_tower_cfg.num_class,
          kernel_regularizer=self._l2_reg,
          name=tower_name + '/output')
      tower_outputs[tower_name] = output_logits

    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict
