# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf

from easy_rec.python.loss.focal_loss import sigmoid_focal_loss_with_logits
from easy_rec.python.loss.jrc_loss import jrc_loss
from easy_rec.python.loss.listwise_loss import listwise_distill_loss
from easy_rec.python.loss.listwise_loss import listwise_rank_loss
from easy_rec.python.loss.pairwise_loss import pairwise_focal_loss
from easy_rec.python.loss.pairwise_loss import pairwise_hinge_loss
from easy_rec.python.loss.pairwise_loss import pairwise_logistic_loss
from easy_rec.python.loss.pairwise_loss import pairwise_loss
from easy_rec.python.protos.loss_pb2 import LossType

from easy_rec.python.loss.zero_inflated_lognormal import zero_inflated_lognormal_loss  # NOQA

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
  elif loss_type == LossType.BINARY_CROSS_ENTROPY_LOSS:
    losses = tf.keras.backend.binary_crossentropy(label, pred, from_logits=True)
    return tf.reduce_mean(losses)
  elif loss_type in [LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS]:
    logging.info('%s is used' % LossType.Name(loss_type))
    return tf.losses.mean_squared_error(
        labels=label, predictions=pred, weights=loss_weight, **kwargs)
  elif loss_type == LossType.ZILN_LOSS:
    loss = zero_inflated_lognormal_loss(label, pred)
    if np.isscalar(loss_weight) and loss_weight != 1.0:
      return loss * loss_weight
    return loss
  elif loss_type == LossType.JRC_LOSS:
    session = kwargs.get('session_ids', None)
    if loss_param is None:
      return jrc_loss(label, pred, session, name=loss_name)
    return jrc_loss(
        label,
        pred,
        session,
        loss_param.alpha,
        loss_weight_strategy=loss_param.loss_weight_strategy,
        sample_weights=loss_weight,
        same_label_loss=loss_param.same_label_loss,
        name=loss_name)
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
    lbl_margin = False if loss_param is None else loss_param.use_label_margin
    return pairwise_logistic_loss(
        label,
        pred,
        session_ids=session,
        temperature=temp,
        hinge_margin=hinge_margin,
        ohem_ratio=ohem_ratio,
        weights=loss_weight,
        use_label_margin=lbl_margin,
        name=loss_name)
  elif loss_type == LossType.PAIRWISE_HINGE_LOSS:
    session = kwargs.get('session_ids', None)
    temp, ohem_ratio, margin = 1.0, 1.0, 1.0
    label_is_logits, use_label_margin, use_exponent = True, True, False
    if loss_param is not None:
      temp = loss_param.temperature
      ohem_ratio = loss_param.ohem_ratio
      margin = loss_param.margin
      label_is_logits = loss_param.label_is_logits
      use_label_margin = loss_param.use_label_margin
      use_exponent = loss_param.use_exponent
    return pairwise_hinge_loss(
        label,
        pred,
        session_ids=session,
        temperature=temp,
        margin=margin,
        ohem_ratio=ohem_ratio,
        weights=loss_weight,
        label_is_logits=label_is_logits,
        use_label_margin=use_label_margin,
        use_exponent=use_exponent,
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
  elif loss_type == LossType.LISTWISE_RANK_LOSS:
    session = kwargs.get('session_ids', None)
    trans_fn, temp, label_is_logits, scale = None, 1.0, False, False
    if loss_param is not None:
      temp = loss_param.temperature
      label_is_logits = loss_param.label_is_logits
      scale = loss_param.scale_logits
      if loss_param.HasField('transform_fn'):
        trans_fn = loss_param.transform_fn
    return listwise_rank_loss(
        label,
        pred,
        session,
        temperature=temp,
        label_is_logits=label_is_logits,
        transform_fn=trans_fn,
        scale_logits=scale,
        weights=loss_weight)
  elif loss_type == LossType.LISTWISE_DISTILL_LOSS:
    session = kwargs.get('session_ids', None)
    trans_fn, temp, label_clip_max_value, scale = None, 1.0, 512.0, False
    if loss_param is not None:
      temp = loss_param.temperature
      label_clip_max_value = loss_param.label_clip_max_value
      scale = loss_param.scale_logits
      if loss_param.HasField('transform_fn'):
        trans_fn = loss_param.transform_fn
    return listwise_distill_loss(
        label,
        pred,
        session,
        temperature=temp,
        label_clip_max_value=label_clip_max_value,
        transform_fn=trans_fn,
        scale_logits=scale,
        weights=loss_weight)
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


def build_kd_loss(kds, prediction_dict, label_dict, feature_dict):
  """Build knowledge distillation loss.

  Args:
    kds: list of knowledge distillation object of type KD.
    prediction_dict: dict of predict_name to predict tensors.
    label_dict: ordered dict of label_name to label tensors.
    feature_dict: dict of feature name to feature value

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

    loss_weight = kd.loss_weight
    if kd.HasField('task_space_indicator_name') and kd.HasField(
        'task_space_indicator_value'):
      in_task_space = tf.to_float(
          tf.equal(feature_dict[kd.task_space_indicator_name],
                   kd.task_space_indicator_value))
      loss_weight = loss_weight * (
          kd.in_task_space_weight * in_task_space + kd.out_task_space_weight *
          (1 - in_task_space))

    label = label_dict[kd.soft_label_name]
    pred = prediction_dict[kd.pred_name]
    epsilon = tf.keras.backend.epsilon()
    num_class = 1 if len(pred.get_shape()) < 2 else pred.get_shape()[-1]

    if kd.loss_type == LossType.BINARY_CROSS_ENTROPY_LOSS:
      if not kd.label_is_logits:  # label is prob
        label = tf.clip_by_value(label, epsilon, 1 - epsilon)
        label = tf.log(label / (1 - label))
      if not kd.pred_is_logits:
        pred = tf.clip_by_value(pred, epsilon, 1 - epsilon)
        pred = tf.log(pred / (1 - pred))
      if kd.temperature > 0:
        label = label / kd.temperature
        pred = pred / kd.temperature
      label = tf.nn.sigmoid(label)  # convert to prob
    elif kd.loss_type == LossType.KL_DIVERGENCE_LOSS:
      if not kd.label_is_logits:  # label is prob
        if num_class == 1:  # for binary classification
          label = tf.clip_by_value(label, epsilon, 1 - epsilon)
          label = tf.log(label / (1 - label))
        else:
          label = tf.math.log(label + epsilon)
          label -= tf.reduce_max(label)
      if not kd.pred_is_logits:
        if num_class == 1:  # for binary classification
          pred = tf.clip_by_value(pred, epsilon, 1 - epsilon)
          pred = tf.log(pred / (1 - pred))
        else:
          pred = tf.math.log(pred + epsilon)
          pred -= tf.reduce_max(pred)
      if kd.temperature > 0:
        label = label / kd.temperature
        pred = pred / kd.temperature
      if num_class > 1:
        label = tf.nn.softmax(label)
        pred = tf.nn.softmax(pred)
      else:
        label = tf.nn.sigmoid(label)  # convert to prob
        pred = tf.nn.sigmoid(pred)  # convert to prob
    elif kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
      if not kd.label_is_logits:
        label = tf.math.log(label + epsilon)
      if not kd.pred_is_logits:
        pred = tf.math.log(pred + epsilon)
      if kd.temperature > 0:
        label = label / kd.temperature
        pred = pred / kd.temperature
      if num_class > 1:
        label = tf.nn.softmax(label)
        pred = tf.nn.softmax(pred)
      elif num_class == 1:
        label = tf.nn.sigmoid(label)
        pred = tf.nn.sigmoid(pred)

    if kd.loss_type == LossType.KL_DIVERGENCE_LOSS:
      if num_class == 1:
        label = tf.expand_dims(label, 1)  # [B, 1]
        labels = tf.concat([1 - label, label], axis=1)  # [B, 2]
        pred = tf.expand_dims(pred, 1)  # [B, 1]
        preds = tf.concat([1 - pred, pred], axis=1)  # [B, 2]
      else:
        labels = label
        preds = pred
      losses = tf.keras.losses.KLD(labels, preds)
      loss_dict[loss_name] = tf.reduce_mean(
          losses, name=loss_name) * loss_weight
    elif kd.loss_type == LossType.BINARY_CROSS_ENTROPY_LOSS:
      losses = tf.keras.backend.binary_crossentropy(
          label, pred, from_logits=True)
      loss_dict[loss_name] = tf.reduce_mean(
          losses, name=loss_name) * loss_weight
    elif kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
      loss_dict[loss_name] = tf.losses.log_loss(
          label, pred, weights=loss_weight)
    elif kd.loss_type == LossType.L2_LOSS:
      loss_dict[loss_name] = tf.losses.mean_squared_error(
          labels=label, predictions=pred, weights=loss_weight)
    else:
      loss_param = kd.WhichOneof('loss_param')
      kwargs = {}
      if loss_param is not None:
        loss_param = getattr(kd, loss_param)
        if hasattr(loss_param, 'session_name'):
          kwargs['session_ids'] = feature_dict[loss_param.session_name]
      loss_dict[loss_name] = build(
          kd.loss_type,
          label,
          pred,
          loss_weight=loss_weight,
          loss_param=loss_param,
          **kwargs)
  return loss_dict
