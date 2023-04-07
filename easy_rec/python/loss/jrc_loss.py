# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def jrc_loss(labels,
             logits,
             session_ids,
             alpha=0.5,
             auto_weight=False,
             name=''):
  """Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model.

     https://arxiv.org/abs/2208.06164

  Args:
    labels: a `Tensor` with shape [batch_size]. e.g. click or not click in the session.
    logits: a `Tensor` with shape [batch_size, 2]. e.g. the value of last neuron before activation.
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    alpha: the weight to balance ranking loss and calibration loss
    auto_weight: bool, whether to learn loss weight between ranking loss and calibration loss
    name: the name of loss
  """
  loss_name = name if name else 'jrc_loss'
  logging.info('[{}] alpha: {}, auto_weight: {}'.format(loss_name, alpha,
                                                        auto_weight))

  ce_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

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

  # Compute list-wise generative loss -log p(x|y, z)
  loss_pos = -tf.reduce_sum(y_pos * tf.nn.log_softmax(l_pos, axis=0), axis=0)
  loss_neg = -tf.reduce_sum(y_neg * tf.nn.log_softmax(l_neg, axis=0), axis=0)
  ge_loss = tf.reduce_mean((loss_pos + loss_neg) / tf.reduce_sum(mask, axis=0))

  # The final JRC model
  if auto_weight:
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
    loss = alpha * ce_loss + (1 - alpha) * ge_loss
  return loss
