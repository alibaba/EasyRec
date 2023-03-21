# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def sigmoid_focal_loss_with_logits(labels,
                                   logits,
                                   gamma=2.0,
                                   alpha=None,
                                   ohem_ratio=1.0,
                                   sample_weights=None,
                                   label_smoothing=0,
                                   name=''):
  """Implements the focal loss function.

  Focal loss was first introduced in the RetinaNet paper
  (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
  classification when you have highly imbalanced classes. It down-weights
  well-classified examples and focuses on hard examples. The loss value is
  much high for a sample which is misclassified by the classifier as compared
  to the loss value corresponding to a well-classified example. One of the
  best use-cases of focal loss is its usage in object detection where the
  imbalance between the background class and other classes is extremely high.

  Args
      labels: `[batch_size]` target integer labels in `{0, 1}`.
      logits: Float `[batch_size]` logits outputs of the network.
      alpha: balancing factor.
      gamma: modulating factor.
      ohem_ratio: the percent of hard examples to be mined
      sample_weights:  Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `losses` dimension).
      label_smoothing: If greater than `0` then smooth the labels.
      name: the name of loss

  Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
      same shape as `y_true`; otherwise, it is scalar.

  Raises:
      ValueError: If the shape of `sample_weight` is invalid or value of
        `gamma` is less than zero
  """
  loss_name = name if name else 'focal_loss'
  assert 0 < ohem_ratio <= 1.0, loss_name + ' ohem_ratio must be in (0, 1]'
  if gamma and gamma < 0:
    raise ValueError('Value of gamma should be greater than or equal to zero')
  logging.info(
      '[{}] gamma: {}, alpha: {}, ohem_ratho: {}, label smoothing: {}'.format(
          loss_name, gamma, alpha, ohem_ratio, label_smoothing))

  y_true = tf.cast(labels, logits.dtype)

  # convert the predictions into probabilities
  y_pred = tf.nn.sigmoid(logits)
  epsilon = 1e-7
  y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
  p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
  weights = tf.pow((1 - p_t), gamma)

  if alpha is not None:
    alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
    weights *= alpha_factor

  if sample_weights is not None:
    if tf.is_numeric_tensor(sample_weights):
      logging.info('[%s] use sample weight' % loss_name)
      weights *= tf.cast(sample_weights, tf.float32)
    elif sample_weights != 1.0:
      logging.info('[%s] use sample weight: %f' % (loss_name, sample_weights))
      weights *= sample_weights

  if ohem_ratio == 1.0:
    return tf.losses.sigmoid_cross_entropy(
        y_true, logits, weights=weights, label_smoothing=label_smoothing)

  losses = tf.losses.sigmoid_cross_entropy(
      y_true,
      logits,
      weights=weights,
      label_smoothing=label_smoothing,
      reduction=tf.losses.Reduction.NONE)
  k = tf.to_float(tf.size(losses)) * tf.convert_to_tensor(ohem_ratio)
  k = tf.to_int32(tf.math.rint(k))
  topk = tf.nn.top_k(losses, k)
  losses = tf.boolean_mask(topk.values, topk.values > 0)
  return tf.reduce_mean(losses)
