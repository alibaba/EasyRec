# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from focal_loss import sigmoid_focal_loss_with_logits
from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss

from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def pairwise_loss(labels, logits, session_ids=None, margin=0, weights=1.0):
  """Pairwise loss.  Also see `pairwise_logistic_loss` below.

  Args:
    labels: a `Tensor` with shape [batch_size]. e.g. click or not click in the session.
    logits: a `Tensor` with shape [batch_size]. e.g. the value of last neuron before activation.
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    margin: the margin between positive and negative sample pair
    weights: sample weights
  """
  logging.info('[pairwise_loss] margin: {}'.format(margin))
  pairwise_logits = tf.math.subtract(
      tf.expand_dims(logits, -1), tf.expand_dims(logits, 0)) - margin
  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0), 0)
  if session_ids is not None:
    logging.info('[pairwise_loss] use session ids')
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)

  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
  pairwise_pseudo_labels = tf.ones_like(pairwise_logits)

  if tf.is_numeric_tensor(weights):
    logging.info('[pairwise_loss] use sample weight')
    weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)
    batch_size, _ = get_shape_list(weights, 2)
    pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))
    pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
  else:
    pairwise_weights = weights

  loss = tf.losses.sigmoid_cross_entropy(
      pairwise_pseudo_labels, pairwise_logits, weights=pairwise_weights)
  # set rank loss to zero if a batch has no positive sample.
  loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
  return loss


def pairwise_focal_loss(labels,
                        logits,
                        session_ids=None,
                        margin=0,
                        gamma=2,
                        alpha=None,
                        weights=1.0):
  logging.info('[pairwise_focal_loss] margin: {}, gamma: {}, alpha: {}'.format(
      margin, gamma, alpha))
  pairwise_logits = tf.math.subtract(
      tf.expand_dims(logits, -1), tf.expand_dims(logits, 0)) - margin
  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0), 0)
  if session_ids is not None:
    logging.info('[pairwise_focal_loss] use session ids')
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)
  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)

  if tf.is_numeric_tensor(weights):
    logging.info('[pairwise_focal_loss] use sample weight')
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
      sample_weights=pairwise_weights)

  # set rank loss to zero if a batch has no positive sample.
  loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
  return loss


def pairwise_logistic_loss(labels,
                           logits,
                           session_ids=None,
                           temperature=1.0,
                           weights=1.0):
  r"""Pairwise logistic loss.

  Definition:
  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))
  $$

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size].
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    temperature: A float number to modify the scores=scores/temperature.
    weights: A scalar, a `Tensor` with shape [batch_size] for each sample
  """
  logits /= temperature
  pairwise_logits = tf.math.subtract(
      tf.expand_dims(logits, -1), tf.expand_dims(logits, 0))

  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0), 0)
  if session_ids is not None:
    logging.info('[pairwise_logistic_loss] use session ids')
    group_equal = tf.equal(
        tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))
    pairwise_mask = tf.logical_and(pairwise_mask, group_equal)
  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)

  # The following is the same as log(1 + exp(-pairwise_logits)).
  losses = tf.nn.relu(-pairwise_logits) + tf.math.log1p(
      tf.exp(-tf.abs(pairwise_logits)))

  if tf.is_numeric_tensor(weights):
    logging.info('[pairwise_logistic_loss] use sample weight')
    weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)
    batch_size, _ = get_shape_list(weights, 2)
    pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))
    pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
  else:
    pairwise_weights = weights

  loss = compute_weighted_loss(losses, pairwise_weights)
  # set rank loss to zero if a batch has no positive sample.
  loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
  return loss
