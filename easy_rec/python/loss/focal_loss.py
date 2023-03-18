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
                                   sample_weights=None):
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
      labels: true targets tensor.
      logits: predictions tensor.
      alpha: balancing factor.
      gamma: modulating factor.

  Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
      same shape as `y_true`; otherwise, it is scalar.

  Raises:
      ValueError: If the shape of `sample_weight` is invalid or value of
        `gamma` is less than zero
  """
  if gamma and gamma < 0:
    raise ValueError('Value of gamma should be greater than or equal to zero')
  logging.info('[focal_loss] gamma: {}, alpha: {}'.format(gamma, alpha))

  y_true = tf.cast(labels, logits.dtype)

  # convert the predictions into probabilities
  y_pred = tf.nn.sigmoid(logits)
  p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
  weights = tf.pow((1 - p_t), gamma)

  if alpha is not None:
    alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
    weights *= alpha_factor

  if sample_weights is not None:
    if tf.is_numeric_tensor(sample_weights):
      weights *= tf.cast(sample_weights, tf.float32)
    else:
      weights *= sample_weights

  return tf.losses.sigmoid_cross_entropy(y_true, logits, weights=weights)
