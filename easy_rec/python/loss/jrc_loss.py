# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def jrc_loss(labels,
             logits,
             session_ids,
             alpha=0.5,
             loss_weight_strategy='fixed',
             sample_weights=1.0,
             same_label_loss=True,
             name=''):
  """Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model.

     https://arxiv.org/abs/2208.06164

  Args:
    labels: a `Tensor` with shape [batch_size]. e.g. click or not click in the session.
    logits: a `Tensor` with shape [batch_size, 2]. e.g. the value of last neuron before activation.
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    alpha: the weight to balance ranking loss and calibration loss
    loss_weight_strategy: str, the loss weight strategy to balancing between ce_loss and ge_loss
    sample_weights: Coefficients for the loss. This must be scalar or broadcastable to
      `labels` (i.e. same rank and each dimension is either 1 or the same).
    same_label_loss: enable ge_loss for sample with same label in a session or not.
    name: the name of loss
  """
  loss_name = name if name else 'jrc_loss'
  logging.info('[{}] alpha: {}, loss_weight_strategy: {}'.format(
      loss_name, alpha, loss_weight_strategy))

  ce_loss = tf.losses.sparse_softmax_cross_entropy(
      labels, logits, weights=sample_weights)

  labels = tf.expand_dims(labels, 1)  # [B, 1]
  labels = tf.concat([1 - labels, labels], axis=1)  # [B, 2]

  batch_size = tf.shape(logits)[0]

  # Mask: shape [B, B], mask[i,j]=1 indicates the i-th sample
  # and j-th sample are in the same context
  mask = tf.equal(
      tf.expand_dims(session_ids, 1), tf.expand_dims(session_ids, 0))
  mask = tf.to_float(mask)

  # Tile logits and label: [B, 2]->[B, B, 2]
  logits = tf.tile(tf.expand_dims(logits, 1), [1, batch_size, 1])
  y = tf.tile(tf.expand_dims(labels, 1), [1, batch_size, 1])

  # Set logits that are not in the same context to -inf
  mask3d = tf.expand_dims(mask, 2)
  y = tf.to_float(y) * mask3d
  logits = logits + (1 - mask3d) * -1e9
  y_neg, y_pos = y[:, :, 0], y[:, :, 1]
  l_neg, l_pos = logits[:, :, 0], logits[:, :, 1]

  if tf.is_numeric_tensor(sample_weights):
    logging.info('[%s] use sample weight' % loss_name)
    weights = tf.expand_dims(tf.cast(sample_weights, tf.float32), 0)
    pairwise_weights = tf.tile(weights, tf.stack([batch_size, 1]))
    y_pos *= pairwise_weights
    y_neg *= pairwise_weights

  # Compute list-wise generative loss -log p(x|y, z)
  if same_label_loss:
    logging.info('[%s] enable same_label_loss' % loss_name)
    loss_pos = -tf.reduce_sum(y_pos * tf.nn.log_softmax(l_pos, axis=0), axis=0)
    loss_neg = -tf.reduce_sum(y_neg * tf.nn.log_softmax(l_neg, axis=0), axis=0)
    ge_loss = tf.reduce_mean(
        (loss_pos + loss_neg) / tf.reduce_sum(mask, axis=0))
  else:
    logging.info('[%s] disable same_label_loss' % loss_name)
    diag = tf.one_hot(tf.range(batch_size), batch_size)
    l_pos = l_pos + (1 - diag) * y_pos * -1e9
    l_neg = l_neg + (1 - diag) * y_neg * -1e9
    loss_pos = -tf.linalg.diag_part(y_pos * tf.nn.log_softmax(l_pos, axis=0))
    loss_neg = -tf.linalg.diag_part(y_neg * tf.nn.log_softmax(l_neg, axis=0))
    ge_loss = tf.reduce_mean(loss_pos + loss_neg)

  tf.summary.scalar('loss/%s_ce' % loss_name, ce_loss)
  tf.summary.scalar('loss/%s_ge' % loss_name, ge_loss)

  # The final JRC model
  if loss_weight_strategy == 'fixed':
    loss = alpha * ce_loss + (1 - alpha) * ge_loss
  elif loss_weight_strategy == 'random_uniform':
    weight = tf.random_uniform([])
    loss = weight * ce_loss + (1 - weight) * ge_loss
    tf.summary.scalar('loss/%s_ce_weight' % loss_name, weight)
    tf.summary.scalar('loss/%s_ge_weight' % loss_name, 1 - weight)
  elif loss_weight_strategy == 'random_normal':
    weights = tf.random_normal([2])
    loss_weight = tf.nn.softmax(weights)
    loss = loss_weight[0] * ce_loss + loss_weight[1] * ge_loss
    tf.summary.scalar('loss/%s_ce_weight' % loss_name, loss_weight[0])
    tf.summary.scalar('loss/%s_ge_weight' % loss_name, loss_weight[1])
  elif loss_weight_strategy == 'random_bernoulli':
    bern = tf.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
    weights = bern.sample(2)
    loss_weight = tf.cond(
        tf.equal(tf.reduce_sum(weights), 1), lambda: weights,
        lambda: tf.convert_to_tensor([0.5, 0.5]))
    loss = loss_weight[0] * ce_loss + loss_weight[1] * ge_loss
    tf.summary.scalar('loss/%s_ce_weight' % loss_name, loss_weight[0])
    tf.summary.scalar('loss/%s_ge_weight' % loss_name, loss_weight[1])
  elif loss_weight_strategy == 'uncertainty':
    uncertainty1 = tf.Variable(
        0, name='%s_ranking_loss_weight' % loss_name, dtype=tf.float32)
    tf.summary.scalar('loss/%s_ranking_uncertainty' % loss_name, uncertainty1)
    uncertainty2 = tf.Variable(
        0, name='%s_calibration_loss_weight' % loss_name, dtype=tf.float32)
    tf.summary.scalar('loss/%s_calibration_uncertainty' % loss_name,
                      uncertainty2)
    loss = tf.exp(-uncertainty1) * ce_loss + 0.5 * uncertainty1
    loss += tf.exp(-uncertainty2) * ge_loss + 0.5 * uncertainty2
  else:
    raise ValueError('Unsupported loss weight strategy `%s` for jrc loss' %
                     loss_weight_strategy)
  if np.isscalar(sample_weights) and sample_weights != 1.0:
    return loss * sample_weights
  return loss
