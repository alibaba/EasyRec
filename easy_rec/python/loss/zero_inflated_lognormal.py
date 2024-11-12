# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Zero-inflated lognormal loss for lifetime value prediction."""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def zero_inflated_lognormal_pred(logits):
  """Calculates predicted mean of zero inflated lognormal logits.

  Arguments:
    logits: [batch_size, 3] tensor of logits.

  Returns:
    positive_probs: [batch_size, 1] tensor of positive probability.
    preds: [batch_size, 1] tensor of predicted mean.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = tf.keras.backend.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = tf.keras.backend.softplus(logits[..., 2:])
  preds = (
      positive_probs *
      tf.keras.backend.exp(loc + 0.5 * tf.keras.backend.square(scale)))
  return positive_probs, preds


def zero_inflated_lognormal_loss(labels, logits, name=''):
  """Computes the zero inflated lognormal loss.

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=zero_inflated_lognormal)
  ```

  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].
    name: the name of loss

  Returns:
    Zero inflated lognormal loss value.
  """
  loss_name = name if name else 'ziln_loss'
  labels = tf.cast(labels, dtype=tf.float32)
  if labels.shape.ndims == 1:
    labels = tf.expand_dims(labels, 1)  # [B, 1]
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  logits.shape.assert_is_compatible_with(
      tf.TensorShape(labels.shape[:-1].as_list() + [3]))

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.backend.binary_crossentropy(
      positive, positive_logits, from_logits=True)
  classification_loss = tf.keras.backend.mean(classification_loss)
  tf.summary.scalar('loss/%s_classify' % loss_name, classification_loss)

  loc = logits[..., 1:2]
  scale = tf.math.maximum(
      tf.keras.backend.softplus(logits[..., 2:]),
      tf.math.sqrt(tf.keras.backend.epsilon()))
  safe_labels = positive * labels + (
      1 - positive) * tf.keras.backend.ones_like(labels)
  regression_loss = -tf.keras.backend.mean(
      positive * tfd.LogNormal(loc=loc, scale=scale).log_prob(safe_labels))
  tf.summary.scalar('loss/%s_regression' % loss_name, regression_loss)
  return classification_loss + regression_loss
