# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss

from easy_rec.python.loss.focal_loss import sigmoid_focal_loss_with_logits
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def pairwise_loss(labels,
                  logits,
                  session_ids=None,
                  margin=0,
                  temperature=1.0,
                  weights=1.0,
                  name=''):
  """Deprecated Pairwise loss.  Also see `pairwise_logistic_loss` below.

  Args:
    labels: a `Tensor` with shape [batch_size]. e.g. click or not click in the session.
    logits: a `Tensor` with shape [batch_size]. e.g. the value of last neuron before activation.
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    margin: the margin between positive and negative sample pair
    temperature: (Optional) The temperature to use for scaling the logits.
    weights: sample weights
    name: the name of loss
  """
  logging.warning(
      'The old `pairwise_loss` is being deprecated. '
      'Please use the new `pairwise_logistic_loss` or `pairwise_focal_loss`')
  loss_name = name if name else 'pairwise_loss'
  logging.info('[{}] margin: {}, temperature: {}'.format(
      loss_name, margin, temperature))

  if temperature != 1.0:
    logits /= temperature
  pairwise_logits = tf.math.subtract(
      tf.expand_dims(logits, -1), tf.expand_dims(logits, 0)) - margin
  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1), tf.expand_dims(labels, 0))
  if session_ids is not None:
    logging.info('[%s] use session ids' % loss_name)
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)

  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
  num_pair = tf.size(pairwise_logits)
  tf.summary.scalar('loss/%s_num_of_pairs' % loss_name, num_pair)

  if tf.is_numeric_tensor(weights):
    logging.info('[%s] use sample weight' % loss_name)
    weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)
    batch_size, _ = get_shape_list(weights, 2)
    pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))
    pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
  else:
    pairwise_weights = weights

  pairwise_pseudo_labels = tf.ones_like(pairwise_logits)
  loss = tf.losses.sigmoid_cross_entropy(
      pairwise_pseudo_labels, pairwise_logits, weights=pairwise_weights)
  # set rank loss to zero if a batch has no positive sample.
  # loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
  return loss


def pairwise_focal_loss(labels,
                        logits,
                        session_ids=None,
                        hinge_margin=None,
                        gamma=2,
                        alpha=None,
                        ohem_ratio=1.0,
                        temperature=1.0,
                        weights=1.0,
                        name=''):
  loss_name = name if name else 'pairwise_focal_loss'
  assert 0 < ohem_ratio <= 1.0, loss_name + ' ohem_ratio must be in (0, 1]'
  logging.info(
      '[{}] hinge margin: {}, gamma: {}, alpha: {}, ohem_ratio: {}, temperature: {}'
      .format(loss_name, hinge_margin, gamma, alpha, ohem_ratio, temperature))

  if temperature != 1.0:
    logits /= temperature
  pairwise_logits = tf.expand_dims(logits, -1) - tf.expand_dims(logits, 0)

  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1), tf.expand_dims(labels, 0))
  if hinge_margin is not None:
    hinge_mask = tf.less(pairwise_logits, hinge_margin)
    pairwise_mask = tf.logical_and(pairwise_mask, hinge_mask)
  if session_ids is not None:
    logging.info('[%s] use session ids' % loss_name)
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)

  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
  num_pair = tf.size(pairwise_logits)
  tf.summary.scalar('loss/%s_num_of_pairs' % loss_name, num_pair)

  if tf.is_numeric_tensor(weights):
    logging.info('[%s] use sample weight' % loss_name)
    weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)
    batch_size, _ = get_shape_list(weights, 2)
    pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))
    pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
  else:
    pairwise_weights = weights

  pairwise_pseudo_labels = tf.ones_like(pairwise_logits)
  loss = sigmoid_focal_loss_with_logits(
      pairwise_pseudo_labels,
      pairwise_logits,
      gamma=gamma,
      alpha=alpha,
      ohem_ratio=ohem_ratio,
      sample_weights=pairwise_weights)
  return loss


def pairwise_logistic_loss(labels,
                           logits,
                           session_ids=None,
                           temperature=1.0,
                           hinge_margin=None,
                           weights=1.0,
                           ohem_ratio=1.0,
                           use_label_margin=False,
                           name=''):
  r"""Computes pairwise logistic loss between `labels` and `logits`, equivalent to RankNet loss.

  Definition:
  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))
  $$

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size].
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each
      sample, used to max GAUC metric. e.g. user_id
    temperature: (Optional) The temperature to use for scaling the logits.
    hinge_margin: the margin between positive and negative logits
    weights: A scalar, a `Tensor` with shape [batch_size] for each sample
    ohem_ratio: the percent of hard examples to be mined
    use_label_margin: whether to use the diff `label[i]-label[j]` as margin
    name: the name of loss
  """
  loss_name = name if name else 'pairwise_logistic_loss'
  assert 0 < ohem_ratio <= 1.0, loss_name + ' ohem_ratio must be in (0, 1]'
  logging.info('[{}] hinge margin: {}, ohem_ratio: {}, temperature: {}'.format(
      loss_name, hinge_margin, ohem_ratio, temperature))

  if temperature != 1.0:
    logits /= temperature
    if use_label_margin:
      labels /= temperature

  pairwise_logits = tf.math.subtract(
      tf.expand_dims(logits, -1), tf.expand_dims(logits, 0))
  if use_label_margin:
    pairwise_logits -= tf.math.subtract(
        tf.expand_dims(labels, -1), tf.expand_dims(labels, 0))
  elif hinge_margin is not None:
    pairwise_logits -= hinge_margin

  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1), tf.expand_dims(labels, 0))
  if session_ids is not None:
    logging.info('[%s] use session ids' % loss_name)
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)

  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
  num_pair = tf.size(pairwise_logits)
  tf.summary.scalar('loss/%s_num_of_pairs' % loss_name, num_pair)

  # The following is the same as log(1 + exp(-pairwise_logits)).
  losses = tf.nn.relu(-pairwise_logits) + tf.math.log1p(
      tf.exp(-tf.abs(pairwise_logits)))

  if tf.is_numeric_tensor(weights):
    logging.info('[%s] use sample weight' % loss_name)
    weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)
    batch_size, _ = get_shape_list(weights, 2)
    pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))
    pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
  else:
    pairwise_weights = weights

  if ohem_ratio == 1.0:
    return compute_weighted_loss(losses, pairwise_weights)

  losses = compute_weighted_loss(
      losses, pairwise_weights, reduction=tf.losses.Reduction.NONE)
  k = tf.to_float(tf.size(losses)) * tf.convert_to_tensor(ohem_ratio)
  k = tf.to_int32(tf.math.rint(k))
  topk = tf.nn.top_k(losses, k)
  losses = tf.boolean_mask(topk.values, topk.values > 0)
  return tf.reduce_mean(losses)


def pairwise_hinge_loss(labels,
                        logits,
                        session_ids=None,
                        temperature=1.0,
                        margin=1.0,
                        weights=1.0,
                        ohem_ratio=1.0,
                        label_is_logits=True,
                        use_label_margin=True,
                        use_exponent=False,
                        name=''):
  r"""Computes pairwise hinge loss between `labels` and `logits`.

  Definition:
  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i \sum_j I[y_i > y_j] \max(0, 1 - (s_i - s_j))
  $$

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size].
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    temperature: (Optional) The temperature to use for scaling the logits.
    margin: the margin between positive and negative logits
    weights: A scalar, a `Tensor` with shape [batch_size] for each sample
    ohem_ratio: the percent of hard examples to be mined
    label_is_logits: Whether `labels` is expected to be a logits tensor.
    use_label_margin: whether to use the diff `label[i]-label[j]` as margin
    use_exponent: whether to use exponential difference
    name: the name of loss
  """
  loss_name = name if name else 'pairwise_hinge_loss'
  assert 0 < ohem_ratio <= 1.0, loss_name + ' ohem_ratio must be in (0, 1]'
  logging.info(
      '[{}] margin: {}, ohem_ratio: {}, temperature: {}, use_exponent: {}, label_is_logits: {}, use_label_margin: {}'
      .format(loss_name, margin, ohem_ratio, temperature, use_exponent,
              label_is_logits, use_label_margin))

  if temperature != 1.0:
    logits /= temperature
    if label_is_logits:
      labels /= temperature
  if use_exponent:
    labels = tf.nn.sigmoid(labels)
    logits = tf.nn.sigmoid(labels)

  pairwise_logits = tf.math.subtract(
      tf.expand_dims(logits, -1), tf.expand_dims(logits, 0))
  pairwise_labels = tf.math.subtract(
      tf.expand_dims(labels, -1), tf.expand_dims(labels, 0))

  pairwise_mask = tf.greater(pairwise_labels, 0)
  if session_ids is not None:
    logging.info('[%s] use session ids' % loss_name)
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)

  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
  pairwise_labels = tf.boolean_mask(pairwise_labels, pairwise_mask)
  num_pair = tf.size(pairwise_logits)
  tf.summary.scalar('loss/%s_num_of_pairs' % loss_name, num_pair)

  if use_label_margin:
    diff = pairwise_labels - pairwise_logits
  else:
    diff = margin - pairwise_logits
  if use_exponent:
    threshold = 88.0  # the max value of float32 is 3.4028235e+38
    safe_diff = tf.clip_by_value(diff, -threshold, threshold)
    losses = tf.nn.relu(tf.exp(safe_diff) - 1.0)
  else:
    losses = tf.nn.relu(diff)

  if tf.is_numeric_tensor(weights):
    logging.info('[%s] use sample weight' % loss_name)
    weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)
    batch_size, _ = get_shape_list(weights, 2)
    pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))
    pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
  else:
    pairwise_weights = weights

  if ohem_ratio == 1.0:
    return compute_weighted_loss(losses, pairwise_weights)

  losses = compute_weighted_loss(
      losses, pairwise_weights, reduction=tf.losses.Reduction.NONE)
  k = tf.to_float(tf.size(losses)) * tf.convert_to_tensor(ohem_ratio)
  k = tf.to_int32(tf.math.rint(k))
  topk = tf.nn.top_k(losses, k)
  losses = tf.boolean_mask(topk.values, topk.values > 0)
  return tf.reduce_mean(losses)
