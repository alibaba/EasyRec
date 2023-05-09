# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.loss.focal_loss import sigmoid_focal_loss_with_logits
from easy_rec.python.loss.jrc_loss import jrc_loss
from easy_rec.python.loss.pairwise_loss import pairwise_focal_loss
from easy_rec.python.loss.pairwise_loss import pairwise_logistic_loss
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
  loss_name = kwargs.pop('loss_name') if 'loss_name' in kwargs else 'unknown'
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
  elif loss_type == LossType.JRC_LOSS:
    alpha = 0.5 if loss_param is None else loss_param.alpha
    auto_weight = False if loss_param is None else not loss_param.HasField(
        'alpha')
    session = kwargs.get('session_ids', None)
    return jrc_loss(
        label, pred, session, alpha, auto_weight=auto_weight, name=loss_name)
  elif loss_type == LossType.PAIR_WISE_LOSS:
    session = kwargs.get('session_ids', None)
    margin = 0 if loss_param is None else loss_param.margin
    temp = 1.0 if loss_param is None else loss_param.temperature
    return pairwise_loss(
        label,
        pred,
        session_ids=session,
        margin=margin,
        temperature=temp,
        weights=loss_weight,
        name=loss_name)
  elif loss_type == LossType.PAIRWISE_LOGISTIC_LOSS:
    session = kwargs.get('session_ids', None)
    temp = 1.0 if loss_param is None else loss_param.temperature
    ohem_ratio = 1.0 if loss_param is None else loss_param.ohem_ratio
    hinge_margin = None
    if loss_param is not None and loss_param.HasField('hinge_margin'):
      hinge_margin = loss_param.hinge_margin
    return pairwise_logistic_loss(
        label,
        pred,
        session_ids=session,
        temperature=temp,
        hinge_margin=hinge_margin,
        ohem_ratio=ohem_ratio,
        weights=loss_weight,
        name=loss_name)
  elif loss_type == LossType.PAIRWISE_FOCAL_LOSS:
    session = kwargs.get('session_ids', None)
    if loss_param is None:
      return pairwise_focal_loss(
          label, pred, session_ids=session, weights=loss_weight, name=loss_name)
    hinge_margin = None
    if loss_param.HasField('hinge_margin'):
      hinge_margin = loss_param.hinge_margin
    return pairwise_focal_loss(
        label,
        pred,
        session_ids=session,
        gamma=loss_param.gamma,
        alpha=loss_param.alpha if loss_param.HasField('alpha') else None,
        hinge_margin=hinge_margin,
        ohem_ratio=loss_param.ohem_ratio,
        temperature=loss_param.temperature,
        weights=loss_weight,
        name=loss_name)
  elif loss_type == LossType.F1_REWEIGHTED_LOSS:
    f1_beta_square = 1.0 if loss_param is None else loss_param.f1_beta_square
    label_smoothing = 0 if loss_param is None else loss_param.label_smoothing
    return f1_reweight_sigmoid_cross_entropy(
        label,
        pred,
        f1_beta_square,
        weights=loss_weight,
        label_smoothing=label_smoothing)
  elif loss_type == LossType.BINARY_FOCAL_LOSS:
    if loss_param is None:
      return sigmoid_focal_loss_with_logits(
          label, pred, sample_weights=loss_weight, name=loss_name)
    gamma = loss_param.gamma
    alpha = None
    if loss_param.HasField('alpha'):
      alpha = loss_param.alpha
    return sigmoid_focal_loss_with_logits(
        label,
        pred,
        gamma=gamma,
        alpha=alpha,
        ohem_ratio=loss_param.ohem_ratio,
        sample_weights=loss_weight,
        label_smoothing=loss_param.label_smoothing,
        name=loss_name)
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
