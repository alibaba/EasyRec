# coding=utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def f1_reweight_sigmoid_cross_entropy(labels,
                                      logits,
                                      beta_square,
                                      label_smoothing=0,
                                      weights=None):
  """Refer paper: Adaptive Scaling for Sparse Detection in Information Extraction."""
  probs = tf.nn.sigmoid(logits)
  if len(logits.shape.as_list()) == 1:
    logits = tf.expand_dims(logits, -1)
  if len(labels.shape.as_list()) == 1:
    labels = tf.expand_dims(labels, -1)
  labels = tf.to_float(labels)
  batch_size = tf.shape(labels)[0]
  batch_size_float = tf.to_float(batch_size)
  num_pos = tf.reduce_sum(labels, axis=0)
  num_neg = batch_size_float - num_pos
  tp = tf.reduce_sum(probs, axis=0)
  tn = batch_size_float - tp
  neg_weight = tp / (beta_square * num_pos + num_neg - tn + 1e-8)
  neg_weight_tile = tf.tile(tf.expand_dims(neg_weight, 0), [batch_size, 1])
  final_weights = tf.where(
      tf.equal(labels, 1.0), tf.ones_like(labels), neg_weight_tile)
  if weights is not None:
    weights = tf.cast(weights, tf.float32)
    if len(weights.shape.as_list()) == 1:
      weights = tf.expand_dims(weights, -1)
    final_weights *= weights
  return tf.losses.sigmoid_cross_entropy(
      labels, logits, final_weights, label_smoothing=label_smoothing)
