# coding=utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def pairwise_loss(logits, labels):
  pairwise_logits = tf.expand_dims(logits, -1) - tf.expand_dims(
      logits, 0)
  logging.info('[pairwise_loss] pairwise logits: {}'.format(pairwise_logits))

  pairwise_mask = tf.greater(
      tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0), 0)
  logging.info('[pairwise_loss] mask: {}'.format(pairwise_mask))

  pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
  logging.info('[pairwise_loss] after masking: {}'.format(pairwise_logits))

  pairwise_pseudo_labels = tf.ones_like(pairwise_logits)
  loss = tf.losses.sigmoid_cross_entropy(
      pairwise_pseudo_labels, pairwise_logits)
  # set rank loss to zero if a batch has no positive sample.
  loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
  return loss
