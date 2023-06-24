# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.protos.esmm_pb2 import ESMM as ESMMConfig
from easy_rec.python.protos.loss_pb2 import LossType

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses


class ESMM(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(ESMM, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert self._model_config.WhichOneof('model') == 'esmm', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.esmm
    assert isinstance(self._model_config, ESMMConfig)

    self._group_num = len(self._model_config.groups)
    self._group_features = []
    if self.has_backbone:
      logging.info('use bottom backbone network')
    elif self._group_num > 0:
      logging.info('group_num: {0}'.format(self._group_num))
      for group_id in range(self._group_num):
        group = self._model_config.groups[group_id]
        group_feature, _ = self._input_layer(self._feature_dict, group.input)
        self._group_features.append(group_feature)
    else:
      group_feature, _ = self._input_layer(self._feature_dict, 'all')
      self._group_features.append(group_feature)

    # This model only supports two tasks (cvr+ctr or playtime+ctr).
    # In order to be consistent with the paper,
    # we call these two towers cvr_tower (main tower) and ctr_tower (aux tower).
    self._cvr_tower_cfg = self._model_config.cvr_tower
    self._ctr_tower_cfg = self._model_config.ctr_tower
    self._init_towers([self._cvr_tower_cfg, self._ctr_tower_cfg])

    assert self._model_config.ctr_tower.loss_type == LossType.CLASSIFICATION, \
        'ctr tower must be binary classification.'
    for task_tower_cfg in self._task_towers:
      assert task_tower_cfg.num_class == 1, 'Does not support multiclass classification problem'

  def build_loss_graph(self):
    """Build loss graph.

    Returns:
      self._loss_dict: Weighted loss of ctr and cvr.
    """
    cvr_tower_name = self._cvr_tower_cfg.tower_name
    ctr_tower_name = self._ctr_tower_cfg.tower_name
    cvr_label_name = self._label_name_dict[cvr_tower_name]
    ctr_label_name = self._label_name_dict[ctr_tower_name]
    if self._cvr_tower_cfg.loss_type == LossType.CLASSIFICATION:
      ctcvr_label = tf.cast(
          self._labels[cvr_label_name] * self._labels[ctr_label_name],
          tf.float32)
      cvr_losses = tf.keras.backend.binary_crossentropy(
          ctcvr_label, self._prediction_dict['probs_ctcvr'])
      cvr_loss = tf.reduce_sum(cvr_losses, name='ctcvr_loss')
      # The weight defaults to 1.
      self._loss_dict['weighted_cross_entropy_loss_%s' %
                      cvr_tower_name] = self._cvr_tower_cfg.weight * cvr_loss

    elif self._cvr_tower_cfg.loss_type == LossType.L2_LOSS:
      logging.info('l2 loss is used')
      cvr_dtype = self._labels[cvr_label_name].dtype
      ctcvr_label = self._labels[cvr_label_name] * tf.cast(
          self._labels[ctr_label_name], cvr_dtype)
      cvr_loss = tf.losses.mean_squared_error(
          labels=ctcvr_label,
          predictions=self._prediction_dict['y_ctcvr'],
          weights=self._sample_weight)
      self._loss_dict['weighted_l2_loss_%s' %
                      cvr_tower_name] = self._cvr_tower_cfg.weight * cvr_loss
    _labels = tf.cast(self._labels[ctr_label_name], tf.float32)
    _logits = self._prediction_dict['logits_%s' % ctr_tower_name]
    cross = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=_labels, logits=_logits, name='ctr_loss')
    ctr_loss = tf.reduce_sum(cross)
    self._loss_dict['weighted_cross_entropy_loss_%s' %
                    ctr_tower_name] = self._ctr_tower_cfg.weight * ctr_loss
    return self._loss_dict

  def build_metric_graph(self, eval_config):
    """Build metric graph.

    Args:
      eval_config: Evaluation configuration.

    Returns:
      metric_dict: Calculate AUC of ctr, cvr and ctrvr.
    """
    metric_dict = {}

    cvr_tower_name = self._cvr_tower_cfg.tower_name
    ctr_tower_name = self._ctr_tower_cfg.tower_name
    cvr_label_name = self._label_name_dict[cvr_tower_name]
    ctr_label_name = self._label_name_dict[ctr_tower_name]
    for metric in self._cvr_tower_cfg.metrics_set:
      # CTCVR metric
      ctcvr_label_name = cvr_label_name + '_ctcvr'
      cvr_dtype = self._labels[cvr_label_name].dtype
      self._labels[ctcvr_label_name] = self._labels[cvr_label_name] * tf.cast(
          self._labels[ctr_label_name], cvr_dtype)
      metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=self._cvr_tower_cfg.loss_type,
              label_name=ctcvr_label_name,
              num_class=self._cvr_tower_cfg.num_class,
              suffix='_ctcvr'))

      # CVR metric
      cvr_label_masked_name = cvr_label_name + '_masked'
      ctr_mask = self._labels[ctr_label_name] > 0
      self._labels[cvr_label_masked_name] = tf.boolean_mask(
          self._labels[cvr_label_name], ctr_mask)
      pred_prefix = 'probs' if self._cvr_tower_cfg.loss_type == LossType.CLASSIFICATION else 'y'
      pred_name = '%s_%s' % (pred_prefix, cvr_tower_name)
      self._prediction_dict[pred_name + '_masked'] = tf.boolean_mask(
          self._prediction_dict[pred_name], ctr_mask)
      metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=self._cvr_tower_cfg.loss_type,
              label_name=cvr_label_masked_name,
              num_class=self._cvr_tower_cfg.num_class,
              suffix='_%s_masked' % cvr_tower_name))

    for metric in self._ctr_tower_cfg.metrics_set:
      # CTR metric
      metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=self._ctr_tower_cfg.loss_type,
              label_name=ctr_label_name,
              num_class=self._ctr_tower_cfg.num_class,
              suffix='_%s' % ctr_tower_name))
    return metric_dict

  def _add_to_prediction_dict(self, output):
    super(ESMM, self)._add_to_prediction_dict(output)
    if self._cvr_tower_cfg.loss_type == LossType.CLASSIFICATION:
      prob = tf.multiply(
          self._prediction_dict['probs_%s' % self._cvr_tower_cfg.tower_name],
          self._prediction_dict['probs_%s' % self._ctr_tower_cfg.tower_name])
      # pctcvr = pctr * pcvr
      self._prediction_dict['probs_ctcvr'] = prob

    else:
      prob = tf.multiply(
          self._prediction_dict['y_%s' % self._cvr_tower_cfg.tower_name],
          self._prediction_dict['probs_%s' % self._ctr_tower_cfg.tower_name])
      # pctcvr = pctr * pcvr
      self._prediction_dict['y_ctcvr'] = prob

  def build_predict_graph(self):
    """Forward function.

    Returns:
      self._prediction_dict: Prediction result of two tasks.
    """
    if self.has_backbone:
      all_fea = self.backbone
    elif self._group_num > 0:
      group_fea_arr = []
      # Both towers share the underlying network.
      for group_id in range(self._group_num):
        group_fea = self._group_features[group_id]
        group = self._model_config.groups[group_id]
        group_name = group.input
        dnn_model = dnn.DNN(group.dnn, self._l2_reg, group_name,
                            self._is_training)
        group_fea = dnn_model(group_fea)
        group_fea_arr.append(group_fea)
      all_fea = tf.concat(group_fea_arr, axis=1)
    else:
      all_fea = self._group_features[0]

    cvr_tower_name = self._cvr_tower_cfg.tower_name
    dnn_model = dnn.DNN(
        self._cvr_tower_cfg.dnn,
        self._l2_reg,
        name=cvr_tower_name,
        is_training=self._is_training)
    cvr_tower_output = dnn_model(all_fea)
    cvr_tower_output = tf.layers.dense(
        inputs=cvr_tower_output,
        units=1,
        kernel_regularizer=self._l2_reg,
        name='%s/dnn_output' % cvr_tower_name)

    ctr_tower_name = self._ctr_tower_cfg.tower_name
    dnn_model = dnn.DNN(
        self._ctr_tower_cfg.dnn,
        self._l2_reg,
        name=ctr_tower_name,
        is_training=self._is_training)
    ctr_tower_output = dnn_model(all_fea)
    ctr_tower_output = tf.layers.dense(
        inputs=ctr_tower_output,
        units=1,
        kernel_regularizer=self._l2_reg,
        name='%s/dnn_output' % ctr_tower_name)

    tower_outputs = {
        cvr_tower_name: cvr_tower_output,
        ctr_tower_name: ctr_tower_output
    }
    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict

  def get_outputs(self):
    """Get model outputs.

    Returns:
      outputs: The list of tensor names output by the model.
    """
    outputs = super(ESMM, self).get_outputs()
    if self._cvr_tower_cfg.loss_type == LossType.CLASSIFICATION:
      outputs.append('probs_ctcvr')
    elif self._cvr_tower_cfg.loss_type == LossType.L2_LOSS:
      outputs.append('y_ctcvr')
    else:
      raise ValueError('invalid cvr_tower loss type: %s' %
                       str(self._cvr_tower_cfg.loss_type))
    return outputs
