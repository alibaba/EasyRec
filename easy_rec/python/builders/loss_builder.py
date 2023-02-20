# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.ops import math_ops

from easy_rec.python.loss.pairwise_loss import pairwise_loss
from easy_rec.python.protos.loss_pb2 import LossType

from easy_rec.python.loss.f1_reweight_loss import f1_reweight_sigmoid_cross_entropy  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def build(loss_type,
          label,
          pred,
          loss_weight=1.0,
          num_class=1,
          loss_param=None,
          **kwargs):
  if loss_type == LossType.CLASSIFICATION:
    if num_class == 1:
      return tf.losses.sigmoid_cross_entropy(
          label, logits=pred, weights=loss_weight, **kwargs)
    else:
      assert label.dtype in [tf.int32, tf.int64], \
          'label.dtype must in [tf.int32, tf.int64] when use sparse_softmax_cross_entropy.'
      return tf.losses.sparse_softmax_cross_entropy(
          labels=label, logits=pred, weights=loss_weight, **kwargs)
  elif loss_type == LossType.CROSS_ENTROPY_LOSS:
    return tf.losses.log_loss(label, pred, weights=loss_weight, **kwargs)
  elif loss_type in [LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS]:
    logging.info('%s is used' % LossType.Name(loss_type))
    return tf.losses.mean_squared_error(
        labels=label, predictions=pred, weights=loss_weight, **kwargs)
  elif loss_type == LossType.PAIR_WISE_LOSS:
    return pairwise_loss(label, pred)
  elif loss_type == LossType.F1_REWEIGHTED_LOSS:
    f1_beta_square = 1.0 if loss_param is None else loss_param.f1_beta_square
    label_smoothing = 0 if loss_param is None else loss_param.label_smoothing
    return f1_reweight_sigmoid_cross_entropy(
        label,
        pred,
        f1_beta_square,
        weights=loss_weight,
        label_smoothing=label_smoothing)
  elif loss_type == LossType.L2_QUANTILE_LOSS:
    bucketize_label = math_ops.bucketize(label, list(
        loss_param.boundaries)) / len(loss_param.boundaries)
    return tf.losses.mean_squared_error(
        labels=bucketize_label, predictions=pred, weights=loss_weight, **kwargs)
  else:
    raise ValueError('unsupported loss type: %s' % LossType.Name(loss_type))


def build_kd_loss(kds, prediction_dict, label_dict):
  """Build knowledge distillation loss.

  Args:
    kds: list of knowledge distillation object of type KD.
    prediction_dict: dict of predict_name to predict tensors.
    label_dict: ordered dict of label_name to label tensors.

  Return:
    knowledge distillation loss will be add to loss_dict with key: kd_loss.
  """
  loss_dict = {}
  for kd in kds:
    assert kd.pred_name in prediction_dict, \
        'invalid predict_name: %s available ones: %s' % (
            kd.pred_name, ','.join(prediction_dict.keys()))

    loss_name = kd.loss_name
    if not loss_name:
      loss_name = 'kd_loss_' + kd.pred_name.replace('/', '_')
      loss_name += '_' + kd.soft_label_name.replace('/', '_')

    label = label_dict[kd.soft_label_name]
    pred = prediction_dict[kd.pred_name]

    if kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
      if not kd.label_is_logits:
        label = tf.math.log(label + 1e-7)
      if not kd.pred_is_logits:
        pred = tf.math.log(pred + 1e-7)

    if kd.temperature > 0 and kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
      label = label / kd.temperature
      pred = pred / kd.temperature

    if kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
      num_class = 1 if len(pred.get_shape()) < 2 else pred.get_shape()[-1]
      if num_class > 1:
        label = tf.nn.softmax(label)
        pred = tf.nn.softmax(pred)
      elif num_class == 1:
        label = tf.nn.sigmoid(label)
        pred = tf.nn.sigmoid(label)

    if kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
      loss_dict[loss_name] = tf.losses.log_loss(
          label, pred, weights=kd.loss_weight)
    elif kd.loss_type == LossType.L2_LOSS:
      loss_dict[loss_name] = tf.losses.mean_squared_error(
          labels=label, predictions=pred, weights=kd.loss_weight)
    else:
      assert False, 'unsupported loss type for kd: %s' % LossType.Name(
          kd.loss_type)
  return loss_dict
