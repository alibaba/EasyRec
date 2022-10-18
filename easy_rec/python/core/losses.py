# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.protos.loss_pb2 import LossReduction

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def sigmoid_focal_cross_entropy(y_true,
                                y_pred,
                                alpha=0.25,
                                gamma=2.0,
                                reduction=LossReduction.MEAN):
  """Implements the focal loss function.

  Focal loss was first introduced in the RetinaNet paper
  (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
  classification when you have highly imbalanced classes. It down-weights
  well-classified examples and focuses on hard examples. The loss value is
  much high for a sample which is misclassified by the classifier as compared
  to the loss value corresponding to a well-classified example. One of the
  best use-cases of focal loss is its usage in object detection where the
  imbalance between the background class and other classes is extremely high.

  Args:
      y_true: true targets tensor.
      y_pred: predictions tensor.
      alpha: balancing factor.
      gamma: modulating factor.
      reduction: loss reduction type.

  Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
      same shape as `y_true`; otherwise, it is scalar.
  """
  if gamma and gamma < 0:
    raise ValueError('Value of gamma should be greater than or equal to zero')

  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, dtype=y_pred.dtype)

  # Get the cross_entropy for each entry
  ce = tf.losses.sigmoid_cross_entropy(
      y_true, y_pred, reduction=tf.losses.Reduction.NONE)

  # If logits are provided then convert the predictions into probabilities
  pred_prob = tf.sigmoid(y_pred)

  p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
  alpha_factor = 1.0
  modulating_factor = 1.0

  if alpha:
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

  if gamma:
    gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
    modulating_factor = tf.pow((1.0 - p_t), gamma)

  # compute the final loss and return
  if reduction == LossReduction.MEAN:
    return tf.reduce_mean(alpha_factor * modulating_factor * ce)
  elif reduction == LossReduction.SUM:
    return tf.reduce_sum(alpha_factor * modulating_factor * ce)
  elif reduction == LossReduction.MEAN_BY_BATCH_SIZE:
    batch_size = tf.to_float(tf.shape(ce)[0])
    return tf.reduce_sum(alpha_factor * modulating_factor * ce) / batch_size
  else:
    raise NotImplementedError
