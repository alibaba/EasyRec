# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.builders import loss_builder
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos import tower_pb2
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
    metric_dict = {}
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      for metric in task_tower_cfg.metrics_set:
        loss_types = {task_tower_cfg.loss_type}
        if len(task_tower_cfg.losses) > 0:
          loss_types = {loss.loss_type for loss in task_tower_cfg.losses}
        metric_dict.update(
            self._build_metric_impl(
                metric,
                loss_type=loss_types,
                label_name=self._label_name_dict[tower_name],
                num_class=task_tower_cfg.num_class,
                suffix='_%s' % tower_name))
    return metric_dict

  def build_loss_graph(self):
    """Build loss graph for multi task model."""
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

      loss_dict = {}
      losses = task_tower_cfg.losses
      if len(losses) == 0:
        loss_dict = self._build_loss_impl(
            task_tower_cfg.loss_type,
            label_name=self._label_name_dict[tower_name],
            loss_weight=loss_weight,
            num_class=task_tower_cfg.num_class,
            suffix='_%s' % tower_name)
      else:
        for loss in losses:
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
          for loss_name, loss_value in loss_ops.items():
            if loss.learn_loss_weight:
              uncertainty = tf.Variable(
                  0, name='%s_loss_weight' % loss_name, dtype=tf.float32)
              tf.summary.scalar('loss/%s_uncertainty' % loss_name, uncertainty)
              if loss.loss_type in {LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS}:
                loss_dict[loss_name] = 0.5 * tf.exp(
                    -uncertainty) * loss_value + 0.5 * uncertainty
              else:
                loss_dict[loss_name] = tf.exp(
                    -uncertainty) * loss_value + 0.5 * uncertainty
            else:
              loss_dict[loss_name] = loss_value * loss.weight

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
          outputs.extend(
              self._get_outputs_impl(
                  loss.loss_type,
                  task_tower_cfg.num_class,
                  suffix='_%s' % tower_name))
    return list(set(outputs))
