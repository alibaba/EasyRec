# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CapsuleLayer:

  def __init__(self, capsule_config, is_training):
    # max_seq_len: max behaviour sequence length(history length)
    self._max_seq_len = capsule_config.max_seq_len
    # max_k: max high capsule number
    self._max_k = capsule_config.max_k
    # high_dim: high capsule vector dimension
    self._high_dim = capsule_config.high_dim
    # number of Expectation-Maximization iterations
    self._num_iters = capsule_config.num_iters
    # routing_logits_scale
    self._routing_logits_scale = capsule_config.routing_logits_scale
    # routing_logits_stddev
    self._routing_logits_stddev = capsule_config.routing_logits_stddev
    # squash power
    self._squash_pow = capsule_config.squash_pow
    # scale ratio
    self._scale_ratio = capsule_config.scale_ratio
    self._const_caps_num = capsule_config.const_caps_num
    self._is_training = is_training

  def squash(self, inputs):
    """Squash inputs over the last dimension."""
    input_norm = tf.reduce_sum(tf.square(inputs), keep_dims=True, axis=-1)
    input_norm_eps = tf.maximum(input_norm, 1e-8)
    scale_factor = tf.pow(input_norm_eps / (1 + input_norm_eps), self._squash_pow) * \
        self._scale_ratio / tf.sqrt(input_norm_eps)
    tf.summary.histogram('capsule/squash_scale_factor', scale_factor)
    return scale_factor * inputs

  def _build_capsule_simi(self, high_capsules, capsule_num):
    high_capsule_mask = tf.sequence_mask(capsule_num,
                                         tf.shape(high_capsules)[1])
    high_capsules = high_capsules * tf.to_float(high_capsule_mask[:, :, None])
    high_capsules = tf.nn.l2_normalize(high_capsules, axis=-1)
    sum_sqr = tf.square(tf.reduce_sum(high_capsules, axis=1))
    sqr_sum = tf.reduce_sum(tf.square(high_capsules), axis=1)
    simi = sum_sqr - sqr_sum

    div = tf.maximum(tf.to_float(capsule_num * (capsule_num - 1)), 1.0)
    simi = tf.reduce_sum(simi, axis=1) / div

    is_multi = tf.to_float(capsule_num > 1)
    avg_simi = tf.reduce_sum((simi + 1) * is_multi) / \
        (2.0 * tf.reduce_sum(is_multi))
    return avg_simi

  def __call__(self, seq_feas, seq_lens):
    """Capsule layer implementation.

    Args:
      seq_feas: tensor of shape batch_size x self._max_seq_len x low_fea_dim(bsd)
      seq_lens: tensor of shape batch_size

    Return:
      high_capsules: tensor of shape batch_size x max_k x high_dim
    """
    # pad or clip to max_seq_len
    seq_feas = tf.cond(
        tf.greater(tf.shape(seq_feas)[1], self._max_seq_len),
        lambda: seq_feas[:, :self._max_seq_len, :], lambda: tf.cond(
            tf.less(tf.shape(seq_feas)[1], self._max_seq_len), lambda: tf.pad(
                seq_feas, [[0, 0], [
                    0, self._max_seq_len - tf.shape(seq_feas)[1]
                ], [0, 0]]), lambda: seq_feas))
    seq_lens = tf.minimum(seq_lens, self._max_seq_len)

    batch_size = tf.shape(seq_lens)[0]
    # max_seq_len x max_num_high_capsule(sh)
    if self._is_training:
      routing_logits = tf.truncated_normal(
          [batch_size, self._max_seq_len, self._max_k],
          stddev=self._routing_logits_stddev)
    else:
      np.random.seed(28)
      routing_logits = tf.constant(
          np.random.uniform(
              high=self._routing_logits_stddev,
              size=[self._max_seq_len, self._max_k]),
          dtype=tf.float32)
      routing_logits = tf.tile(routing_logits[None, :, :], [batch_size, 1, 1])
    routing_logits = tf.stop_gradient(routing_logits)
    # batch_size x max_seq_len x max_k(bsh)
    low_fea_dim = seq_feas.get_shape()[-1]
    # map low capsule features to high capsule features:
    #    low_fea_dim x high_dim(de)
    bilinear_matrix = tf.get_variable(
        dtype=tf.float32, shape=[low_fea_dim, self._high_dim], name='capsule/S')
    # map sequence feature to high dimensional space
    seq_feas_high = tf.tensordot(seq_feas, bilinear_matrix, axes=1)
    seq_feas_high_stop = tf.stop_gradient(seq_feas_high)
    seq_feas_high_norm = tf.nn.l2_normalize(seq_feas_high_stop, -1)

    if self._const_caps_num:
      logging.info('will use constant number of capsules: %d' % self._max_k)
      num_high_capsules = tf.zeros_like(seq_lens, dtype=tf.int32) + self._max_k
    else:
      logging.info(
          'will use log(seq_len) number of capsules, max_capsules: %d' %
          self._max_k)
      num_high_capsules = tf.maximum(
          1, tf.minimum(self._max_k,
                        tf.to_int32(tf.log(tf.to_float(seq_lens)))))

    # batch_size x max_seq_len(bs)
    mask = tf.sequence_mask(seq_lens, self._max_seq_len)
    mask = tf.cast(mask, tf.float32)
    # batch_size x max_k(bh)
    mask_cap = tf.sequence_mask(num_high_capsules, self._max_k)
    mask_cap = tf.cast(mask_cap, tf.float32)
    # batch_size x max_seq_len x 1(bs1)
    # max_seq_thresh = (mask[:, :, None] * 2 - 1) * 1e32
    # batch_size x 1 x h (b1h)
    max_cap_thresh = (tf.cast(mask_cap[:, None, :], tf.float32) * 2 - 1) * 1e32
    for iter_id in range(self._num_iters):
      # batch_size x max_seq_len x max_k(bsh)
      routing_logits = tf.minimum(routing_logits, max_cap_thresh)
      routing_logits = tf.nn.softmax(routing_logits, axis=2)

      routing_logits = routing_logits * mask[:, :, None]

      logits_simi = self._build_capsule_simi(routing_logits, seq_lens)
      tf.summary.scalar('capsule/rlogits_simi_%d' % iter_id, logits_simi)

      seq_fea_simi = self._build_capsule_simi(seq_feas_high_stop, seq_lens)
      tf.summary.scalar('capsule/seq_fea_simi_%d' % iter_id, seq_fea_simi)

      # batch_size x max_k x high_dim(bse,bsh->bhe)
      high_capsules = tf.einsum(
          'bse, bsh->bhe', seq_feas_high_stop
          if iter_id + 1 < self._num_iters else seq_feas_high, routing_logits)
      if iter_id + 1 == self._num_iters:
        capsule_simi = self._build_capsule_simi(high_capsules,
                                                num_high_capsules)
        tf.summary.scalar('caspule/simi_%d' % iter_id, capsule_simi)
        tf.summary.scalar('capsule/before_squash',
                          tf.reduce_mean(tf.norm(high_capsules, axis=-1)))
        high_capsules = self.squash(high_capsules)
        tf.summary.scalar('capsule/after_squash',
                          tf.reduce_mean(tf.norm(high_capsules, axis=-1)))
        capsule_simi_final = self._build_capsule_simi(high_capsules,
                                                      num_high_capsules)
        tf.summary.scalar('caspule/simi_final', capsule_simi_final)
        break

      # batch_size x max_k x high_dim(bhe)
      high_capsules = tf.nn.l2_normalize(high_capsules, -1)
      capsule_simi = self._build_capsule_simi(high_capsules, num_high_capsules)
      tf.summary.scalar('caspule/simi_%d' % iter_id, capsule_simi)
      # batch_size x max_seq_len x max_k(bse, bhe->bsh)
      if self._routing_logits_scale > 0:
        if iter_id == 0:
          logging.info('routing_logits_scale = %.2f' %
                       self._routing_logits_scale)
        routing_logits = tf.einsum('bse, bhe->bsh', seq_feas_high_norm,
                                   high_capsules) * self._routing_logits_scale
      else:
        routing_logits = tf.einsum('bse, bhe->bsh', seq_feas_high_stop,
                                   high_capsules)

    # zero paddings
    high_capsule_mask = tf.sequence_mask(num_high_capsules, self._max_k)
    high_capsules = high_capsules * tf.to_float(high_capsule_mask[:, :, None])
    return high_capsules, num_high_capsules
