# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Zero-inflated lognormal loss for lifetime value prediction."""
import logging
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def log_sigmoid(x):
  return -tf.nn.softplus(-x)  # 兼容 TF 1.12


def zero_inflated_lognormal_pred(
  logits, max_sigma=5.0, max_log_clip=20.0, return_log=False
):
  """Calculates predicted mean of zero inflated lognormal logits.

  Arguments:
    logits: [batch_size, 3] tensor of logits.
    max_sigma: max value of sigma
    return_log: whether return log value
    max_log_clip: max clip value of log space

  Returns:
    positive_probs: [batch_size, 1] tensor of positive probability.
    preds: [batch_size, 1] tensor of predicted mean.
  """
  logging.info('max_sigma=%f, max_log_clip=%f' % (max_sigma, max_log_clip))
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = tf.keras.backend.sigmoid(logits[..., :1])
  log_positive_probs = log_sigmoid(logits[..., :1])
  mu = logits[..., 1:2]
  sigma = tf.keras.backend.softplus(logits[..., 2:])
  sigma = tf.clip_by_value(
    sigma, tf.math.sqrt(tf.keras.backend.epsilon()), max_sigma
  )
  log_mean_pos = mu + 0.5 * tf.keras.backend.square(sigma)
  log_preds = log_positive_probs + log_mean_pos
  if return_log:
    logging.info('return_log=true')
    return positive_probs, log_preds
  else:
    log_preds = tf.clip_by_value(log_preds, -100.0, max_log_clip)
    preds = tf.keras.backend.exp(log_preds)
    return positive_probs, preds


def zero_inflated_lognormal_loss(
  labels,
  logits,
  max_sigma=5.0,
  mu_reg=0.01,
  sigma_reg=0.01,
  class_weight=1.0,
  reg_weight=1.0,
  name=''
):
  """Computes the zero inflated lognormal loss.

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=zero_inflated_lognormal)
  ```

  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].
    max_sigma: max value of sigma
    mu_reg: regularize coefficient of mu
    sigma_reg: regularize coefficient of sigma
    class_weight: weight of classification loss
    reg_weight: weight of regression loss
    name: the name of loss

  Returns:
    Zero inflated lognormal loss value.
  """
  loss_name = name if name else 'ziln_loss'
  logging.info(
    '%s max_sigma=%f, mu_reg=%f, sigma_reg=%f, classify weight:%f, regression weight %f'
    % (loss_name, max_sigma, mu_reg, sigma_reg, class_weight, reg_weight)
  )
  labels = tf.cast(labels, dtype=tf.float32)
  if labels.shape.ndims == 1:
    labels = tf.expand_dims(labels, 1)  # [B, 1]
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  logits.shape.assert_is_compatible_with(
    tf.TensorShape(labels.shape[:-1].as_list() + [3])
  )

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.backend.binary_crossentropy(
    positive, positive_logits, from_logits=True
  )
  classification_loss = tf.keras.backend.mean(classification_loss)
  tf.summary.scalar('loss/%s_classify' % loss_name, classification_loss)

  mu = logits[..., 1:2]
  sigma = tf.keras.backend.softplus(logits[..., 2:])
  sigma = tf.clip_by_value(
    sigma, tf.math.sqrt(tf.keras.backend.epsilon()), max_sigma
  )

  safe_labels = positive * labels + (1 - positive
                                    ) * tf.keras.backend.ones_like(labels)
  logprob = tfd.LogNormal(loc=mu, scale=sigma).log_prob(safe_labels)
  num_pos = tf.reduce_sum(positive) + 1e-8
  regression_loss = -(tf.reduce_sum(positive * logprob) / num_pos)
  tf.summary.scalar('loss/%s_regression' % loss_name, regression_loss)

  # add regular terms
  loc_penalty = mu_reg * tf.reduce_mean(tf.square(mu))
  scale_penalty = sigma_reg * tf.reduce_mean(tf.square(logits[..., 2:]))
  tf.summary.scalar('loss/%s_loc_penalty' % loss_name, loc_penalty)
  tf.summary.scalar('loss/%s_scale_penalty' % loss_name, scale_penalty)

  total_loss = class_weight * classification_loss + reg_weight * regression_loss
  return total_loss + loc_penalty + scale_penalty
