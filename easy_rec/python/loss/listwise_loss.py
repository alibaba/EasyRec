# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.utils.load_class import load_by_path


def _list_wise_loss(x, labels, logits, session_ids, label_is_logits):
  mask = tf.equal(x, session_ids)
  logits = tf.boolean_mask(logits, mask)
  labels = tf.boolean_mask(labels, mask)
  y = tf.nn.softmax(labels) if label_is_logits else labels
  y_hat = tf.nn.log_softmax(logits)
  return -tf.reduce_sum(y * y_hat)


def _list_prob_loss(x, labels, logits, session_ids):
  mask = tf.equal(x, session_ids)
  logits = tf.boolean_mask(logits, mask)
  labels = tf.boolean_mask(labels, mask)
  y = labels / tf.reduce_sum(labels)
  y_hat = tf.nn.log_softmax(logits)
  return -tf.reduce_sum(y * y_hat)


def listwise_rank_loss(labels,
                       logits,
                       session_ids,
                       transform_fn=None,
                       temperature=1.0,
                       label_is_logits=False,
                       scale_logits=False,
                       weights=1.0,
                       name='listwise_loss'):
  r"""Computes listwise softmax cross entropy loss between `labels` and `logits`.

  Definition:
  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i y_j \log( \frac{\exp(s_i)}{\sum_j exp(s_j)} )
  $$

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size].
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    transform_fn: an affine transformation function of labels
    temperature: (Optional) The temperature to use for scaling the logits.
    label_is_logits: Whether `labels` is expected to be a logits tensor.
          By default, we consider that `labels` encodes a probability distribution.
    scale_logits: Whether to scale the logits.
    weights: sample weights
    name: the name of loss
  """
  loss_name = name if name else 'listwise_rank_loss'
  logging.info('[{}] temperature: {}, scale logits: {}'.format(
      loss_name, temperature, scale_logits))
  labels = tf.to_float(labels)
  if scale_logits:
    with tf.variable_scope(loss_name):
      w = tf.get_variable(
          'scale_w',
          dtype=tf.float32,
          shape=(1,),
          initializer=tf.ones_initializer())
      b = tf.get_variable(
          'scale_b',
          dtype=tf.float32,
          shape=(1,),
          initializer=tf.zeros_initializer())
    logits = logits * tf.abs(w) + b
  if temperature != 1.0:
    logits /= temperature
    if label_is_logits:
      labels /= temperature
  if transform_fn is not None:
    trans_fn = load_by_path(transform_fn)
    labels = trans_fn(labels)

  sessions, _ = tf.unique(tf.squeeze(session_ids))
  tf.summary.scalar('loss/%s_num_of_group' % loss_name, tf.size(sessions))
  losses = tf.map_fn(
      lambda x: _list_wise_loss(x, labels, logits, session_ids, label_is_logits
                                ),
      sessions,
      dtype=tf.float32)
  if tf.is_numeric_tensor(weights):
    logging.error('[%s] use unsupported sample weight' % loss_name)
    return tf.reduce_mean(losses)
  else:
    return tf.reduce_mean(losses) * weights


def listwise_distill_loss(labels,
                          logits,
                          session_ids,
                          transform_fn=None,
                          temperature=1.0,
                          label_clip_max_value=512,
                          scale_logits=False,
                          weights=1.0,
                          name='listwise_distill_loss'):
  r"""Computes listwise softmax cross entropy loss between `labels` and `logits`.

  Definition:
  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i y_j \log( \frac{\exp(s_i)}{\sum_j exp(s_j)} )
  $$

  Args:
    labels: A `Tensor` of the same shape as `logits` representing the rank position of a base model.
    logits: A `Tensor` with shape [batch_size].
    session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
    transform_fn: an transformation function of labels.
    temperature: (Optional) The temperature to use for scaling the logits.
    label_clip_max_value: clip the labels to this value.
    scale_logits: Whether to scale the logits.
    weights: sample weights
    name: the name of loss
  """
  loss_name = name if name else 'listwise_rank_loss'
  logging.info('[{}] temperature: {}'.format(loss_name, temperature))
  labels = tf.to_float(labels)  # supposed to be positions of a teacher model
  labels = tf.clip_by_value(labels, 1, label_clip_max_value)
  if transform_fn is not None:
    trans_fn = load_by_path(transform_fn)
    labels = trans_fn(labels)
  else:
    labels = tf.log1p(label_clip_max_value) - tf.log(labels)

  if scale_logits:
    with tf.variable_scope(loss_name):
      w = tf.get_variable(
          'scale_w',
          dtype=tf.float32,
          shape=(1,),
          initializer=tf.ones_initializer())
      b = tf.get_variable(
          'scale_b',
          dtype=tf.float32,
          shape=(1,),
          initializer=tf.zeros_initializer())
    logits = logits * tf.abs(w) + b
  if temperature != 1.0:
    logits /= temperature

  sessions, _ = tf.unique(tf.squeeze(session_ids))
  tf.summary.scalar('loss/%s_num_of_group' % loss_name, tf.size(sessions))
  losses = tf.map_fn(
      lambda x: _list_prob_loss(x, labels, logits, session_ids),
      sessions,
      dtype=tf.float32)
  if tf.is_numeric_tensor(weights):
    logging.error('[%s] use unsupported sample weight' % loss_name)
    return tf.reduce_mean(losses)
  else:
    return tf.reduce_mean(losses) * weights
