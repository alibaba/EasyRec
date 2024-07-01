# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from collections import OrderedDict

import tensorflow as tf
from google.protobuf import struct_pb2
from tensorflow.python.keras.layers import Dense

from easy_rec.python.builders import loss_builder
from easy_rec.python.layers.dnn import DNN
from easy_rec.python.layers.keras.attention import Attention
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos import tower_pb2
from easy_rec.python.protos.easy_rec_model_pb2 import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MultiTaskModel(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MultiTaskModel, self).__init__(model_config, feature_configs,
                                         features, labels, is_training)
    self._task_towers = []
    self._task_num = None
    self._label_name_dict = {}

  def build_predict_graph(self):
    if not self.has_backbone:
      raise NotImplementedError(
          'method `build_predict_graph` must be implemented when backbone network do not exists'
      )
    model = self._model_config.WhichOneof('model')
    assert model == 'model_params', '`model_params` must be configured'
    config = self._model_config.model_params

    self._init_towers(config.task_towers)

    backbone = self.backbone
    if type(backbone) in (list, tuple):
      if len(backbone) != len(config.task_towers):
        raise ValueError(
            'The number of backbone outputs and task towers must be equal')
      task_input_list = backbone
    else:
      task_input_list = [backbone] * len(config.task_towers)

    tower_features = {}
    for i, task_tower_cfg in enumerate(config.task_towers):
      tower_name = task_tower_cfg.tower_name
      if task_tower_cfg.HasField('dnn'):
        tower_dnn = DNN(
            task_tower_cfg.dnn,
            self._l2_reg,
            name=tower_name,
            is_training=self._is_training)
        tower_output = tower_dnn(task_input_list[i])
      else:
        tower_output = task_input_list[i]
      tower_features[tower_name] = tower_output

    tower_outputs = {}
    relation_features = {}
    # bayes network
    for task_tower_cfg in config.task_towers:
      tower_name = task_tower_cfg.tower_name
      if task_tower_cfg.HasField('relation_dnn'):
        relation_dnn = DNN(
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
      elif task_tower_cfg.use_ait_module:
        tower_inputs = [tower_features[tower_name]]
        for relation_tower_name in task_tower_cfg.relation_tower_names:
          tower_inputs.append(relation_features[relation_tower_name])
        if len(tower_inputs) == 1:
          relation_fea = tower_inputs[0]
          relation_features[tower_name] = relation_fea
        else:
          if task_tower_cfg.HasField('ait_project_dim'):
            dim = task_tower_cfg.ait_project_dim
          else:
            dim = int(tower_inputs[0].shape[-1])
          queries = tf.stack([Dense(dim)(x) for x in tower_inputs], axis=1)
          keys = tf.stack([Dense(dim)(x) for x in tower_inputs], axis=1)
          values = tf.stack([Dense(dim)(x) for x in tower_inputs], axis=1)
          st_params = struct_pb2.Struct()
          st_params.update({'scale_by_dim': True})
          params = Parameter(st_params, True)
          attention_layer = Attention(params, name='AITM_%s' % tower_name)
          result = attention_layer([queries, values, keys])
          relation_fea = result[:, 0, :]
          relation_features[tower_name] = relation_fea
      else:
        relation_fea = tower_features[tower_name]

      output_logits = tf.layers.dense(
          relation_fea,
          task_tower_cfg.num_class,
          kernel_regularizer=self._l2_reg,
          name=tower_name + '/output')
      tower_outputs[tower_name] = output_logits

    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict

  def _init_towers(self, task_tower_configs):
    """Init task towers."""
    self._task_towers = task_tower_configs
    self._task_num = len(task_tower_configs)
    for i, task_tower_config in enumerate(task_tower_configs):
      assert isinstance(task_tower_config, tower_pb2.TaskTower) or \
          isinstance(task_tower_config, tower_pb2.BayesTaskTower), \
          'task_tower_config must be a instance of tower_pb2.TaskTower or tower_pb2.BayesTaskTower'
      tower_name = task_tower_config.tower_name

      # For label backward compatibility with list
      if self._labels is not None:
        if task_tower_config.HasField('label_name'):
          label_name = task_tower_config.label_name
        else:
          # If label name is not specified, task_tower and label will be matched by order
          label_name = list(self._labels.keys())[i]
          logging.info('Task Tower [%s] use label [%s]' %
                       (tower_name, label_name))
        assert label_name in self._labels, 'label [%s] must exists in labels' % label_name
        self._label_name_dict[tower_name] = label_name

  def _add_to_prediction_dict(self, output):
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      if len(task_tower_cfg.losses) == 0:
        self._prediction_dict.update(
            self._output_to_prediction_impl(
                output[tower_name],
                loss_type=task_tower_cfg.loss_type,
                num_class=task_tower_cfg.num_class,
                suffix='_%s' % tower_name))
      else:
        for loss in task_tower_cfg.losses:
          self._prediction_dict.update(
              self._output_to_prediction_impl(
                  output[tower_name],
                  loss_type=loss.loss_type,
                  num_class=task_tower_cfg.num_class,
                  suffix='_%s' % tower_name))

  def build_metric_graph(self, eval_config):
    """Build metric graph for multi task model."""
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      for metric in task_tower_cfg.metrics_set:
        loss_types = {task_tower_cfg.loss_type}
        if len(task_tower_cfg.losses) > 0:
          loss_types = {loss.loss_type for loss in task_tower_cfg.losses}
        self._metric_dict.update(
            self._build_metric_impl(
                metric,
                loss_type=loss_types,
                label_name=self._label_name_dict[tower_name],
                num_class=task_tower_cfg.num_class,
                suffix='_%s' % tower_name))
    return self._metric_dict

  def build_loss_weight(self):
    loss_weights = OrderedDict()
    num_loss = 0
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      losses = task_tower_cfg.losses
      n = len(losses)
      if n > 0:
        loss_weights[tower_name] = [loss.weight for loss in losses]
        num_loss += n
      else:
        loss_weights[tower_name] = [1.0]
        num_loss += 1

    strategy = self._base_model_config.loss_weight_strategy
    if strategy == self._base_model_config.Random:
      weights = tf.random_normal([num_loss])
      weights = tf.nn.softmax(weights)
      i = 0
      for k, v in loss_weights.items():
        n = len(v)
        loss_weights[k] = weights[i:i + n]
        i += n
    return loss_weights

  def get_learnt_loss(self, loss_type, name, value):
    strategy = self._base_model_config.loss_weight_strategy
    if strategy == self._base_model_config.Uncertainty:
      uncertainty = tf.Variable(
          0, name='%s_loss_weight' % name, dtype=tf.float32)
      tf.summary.scalar('loss/%s_uncertainty' % name, uncertainty)
      if loss_type in {LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS}:
        return 0.5 * tf.exp(-uncertainty) * value + 0.5 * uncertainty
      else:
        return tf.exp(-uncertainty) * value + 0.5 * uncertainty
    else:
      strategy_name = EasyRecModel.LossWeightStrategy.Name(strategy)
      raise ValueError('Unsupported loss weight strategy: ' + strategy_name)

  def build_loss_graph(self):
    """Build loss graph for multi task model."""
    task_loss_weights = self.build_loss_weight()
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      loss_weight = task_tower_cfg.weight
      if task_tower_cfg.use_sample_weight:
        loss_weight *= self._sample_weight

      if hasattr(task_tower_cfg, 'task_space_indicator_label') and \
          task_tower_cfg.HasField('task_space_indicator_label'):
        in_task_space = tf.to_float(
            self._labels[task_tower_cfg.task_space_indicator_label] > 0)
        loss_weight = loss_weight * (
            task_tower_cfg.in_task_space_weight * in_task_space +
            task_tower_cfg.out_task_space_weight * (1 - in_task_space))

      task_loss_weight = task_loss_weights[tower_name]
      loss_dict = {}
      losses = task_tower_cfg.losses
      if len(losses) == 0:
        loss_dict = self._build_loss_impl(
            task_tower_cfg.loss_type,
            label_name=self._label_name_dict[tower_name],
            loss_weight=loss_weight,
            num_class=task_tower_cfg.num_class,
            suffix='_%s' % tower_name)
        for loss_name in loss_dict.keys():
          loss_dict[loss_name] = loss_dict[loss_name] * task_loss_weight[0]
      else:
        calibrate_loss = []
        for loss in losses:
          if loss.loss_type == LossType.ORDER_CALIBRATE_LOSS:
            y_t = self._prediction_dict['probs_%s' % tower_name]
            for relation_tower_name in task_tower_cfg.relation_tower_names:
              y_rt = self._prediction_dict['probs_%s' % relation_tower_name]
              cali_loss = tf.reduce_mean(tf.nn.relu(y_t - y_rt))
              calibrate_loss.append(cali_loss * loss.weight)
              logging.info('calibrate loss: %s -> %s' %
                           (relation_tower_name, tower_name))
            continue
          loss_param = loss.WhichOneof('loss_param')
          if loss_param is not None:
            loss_param = getattr(loss, loss_param)
          loss_ops = self._build_loss_impl(
              loss.loss_type,
              label_name=self._label_name_dict[tower_name],
              loss_weight=loss_weight,
              num_class=task_tower_cfg.num_class,
              suffix='_%s' % tower_name,
              loss_name=loss.loss_name,
              loss_param=loss_param)
          for i, loss_name in enumerate(loss_ops):
            loss_value = loss_ops[loss_name]
            if loss.learn_loss_weight:
              loss_dict[loss_name] = self.get_learnt_loss(
                  loss.loss_type, loss_name, loss_value)
            else:
              loss_dict[loss_name] = loss_value * task_loss_weight[i]
        if calibrate_loss:
          cali_loss = tf.add_n(calibrate_loss)
          loss_dict['order_calibrate_loss'] = cali_loss
          tf.summary.scalar('loss/order_calibrate_loss', cali_loss)
      self._loss_dict.update(loss_dict)

    kd_loss_dict = loss_builder.build_kd_loss(self.kd, self._prediction_dict,
                                              self._labels)
    self._loss_dict.update(kd_loss_dict)

    return self._loss_dict

  def get_outputs(self):
    outputs = []
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      if len(task_tower_cfg.losses) == 0:
        outputs.extend(
            self._get_outputs_impl(
                task_tower_cfg.loss_type,
                task_tower_cfg.num_class,
                suffix='_%s' % tower_name))
      else:
        for loss in task_tower_cfg.losses:
          if loss.loss_type == LossType.ORDER_CALIBRATE_LOSS:
            continue
          outputs.extend(
              self._get_outputs_impl(
                  loss.loss_type,
                  task_tower_cfg.num_class,
                  suffix='_%s' % tower_name))
    return list(set(outputs))
