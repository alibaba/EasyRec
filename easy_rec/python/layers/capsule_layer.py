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
    self._is_training = is_training

  def squash(self, inputs):
    """Squash inputs over the last dimension."""
    input_norm = tf.reduce_sum(tf.square(inputs), keep_dims=True, axis=-1)
    scalar_factor = input_norm / (1 + input_norm) / tf.sqrt(input_norm + 1e-8)
    return scalar_factor * inputs

  def __call__(self, seq_feas, seq_lens, debug_feas):
    """Capsule layer.

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
    num_high_capsules = tf.maximum(
        1, tf.minimum(self._max_k, tf.to_int32(tf.log(tf.to_float(seq_lens)))))
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
      # batch_size x max_k x high_dim(bse,bsh->bhe)
      high_capsules = tf.einsum(
          'bse, bsh->bhe', seq_feas_high_stop
          if iter_id + 1 < self._num_iters else seq_feas_high, routing_logits)
      if iter_id + 1 == self._num_iters:
        tmp_ids = tf.where(num_high_capsules > 1)
        tmp_ids = tf.squeeze(tmp_ids, axis=1)
        tmp_ids = tmp_ids[:3]
        # tmp_logits = tf.gather(routing_logits, tmp_ids)
        # tmp_seq_lens = tf.gather(seq_lens, tmp_ids)
        # tmp_num_high = tf.gather(num_high_capsules, tmp_ids)

        # tmp_debug_feas = tf.gather(debug_feas, tmp_ids)

        # def _get_simi(tmp_high_capsules, tmp_num_high):
        #   simi_arr = []
        #   for i in range(len(tmp_num_high)):
        #     tmp_fea = tmp_high_capsules[i]
        #     tmp_fea_len = np.sqrt(np.sum(tmp_fea * tmp_fea, axis=1)) + 1e-10
        #     tmp_fea = tmp_fea / tmp_fea_len[:, None]
        #     tmp_len = tmp_num_high[i]
        #     for m in range(tmp_len):
        #       for n in range(m+1, tmp_len):
        #         simi_arr.append(np.sum(tmp_fea[m]*tmp_fea[n]))    
        #   return np.array([np.mean(simi_arr), np.min(simi_arr), simi_arr[0]], dtype=np.float32)

        # def _get_min_max(tmp_logits, tmp_seq_lens):
        #   min_max = []
        #   msgs = []
        #   def _vec2str(x):
        #     return '[' + ','.join(['%.3f' % k for k in x]) + ']'
        #   for i in range(len(tmp_seq_lens)):
        #     tmp_len = tmp_seq_lens[i]
        #     tmp_logit = tmp_logits[i][:tmp_len]
        #     min_max.append([np.min(tmp_logit), np.max(tmp_logit), tmp_len])
        #     tmp_logit = np.unique([_vec2str(x) for x in tmp_logit])
        #     msgs.append('Find [%d] logit: %s' % (len(tmp_logit), ','.join(tmp_logit)))
        #   return np.array(min_max, dtype=np.float32), msgs

        # # seq_feas
        # tmp_seq_feas = tf.gather(seq_feas, tmp_ids)
        # def _get_seq_diff(tmp_seq_feas, tmp_seq_lens):
        #   diff_arr = []
        #   for i in range(len(tmp_seq_lens)):
        #     tmp_len_i = tmp_seq_lens[i]
        #     tmp_seq_fea_i = tmp_seq_feas[i]
        #     for j in range(i+1, len(tmp_seq_lens)):
        #       tmp_len_j = tmp_seq_lens[j]
        #       tmp_seq_fea_j = tmp_seq_feas[j]
        #       if tmp_len_i != tmp_len_j:
        #         diff_arr.append('%d_%d:%d_%d' % (i,j, tmp_len_i, tmp_len_j))
        #       else: 
        #         diff_arr.append('%d_%d:%d_%.3f' % (i,j, tmp_len_i, 
        #            np.sum(np.abs(tmp_seq_fea_i[:tmp_len_i] - tmp_seq_fea_j[:tmp_len_j]))))
        #   return [diff_arr]
        #     
        # py_min_max, py_msgs = tf.py_func(_get_min_max, [tmp_logits, tmp_seq_lens],
        #     Tout=[tf.float32, tf.string])
        # py_seq_diff = tf.py_func(_get_seq_diff, [tmp_seq_feas, tmp_seq_lens],
        #     Tout=tf.string)
        # tmp_high_capsules = tf.gather(high_capsules, tmp_ids)
        # py_simi = tf.py_func(_get_simi, [tmp_high_capsules, tmp_num_high], Tout=tf.float32)
        high_capsules = self.squash(high_capsules)
        # tmp_high_capsules = tf.gather(high_capsules, tmp_ids)
        # py_simi_squash = tf.py_func(_get_simi, [tmp_high_capsules, tmp_num_high], Tout=tf.float32)
        # high_capsules = tf.Print(high_capsules, [tmp_logits[:, :10, :],
        #     tf.shape(routing_logits), 'tmp_high_capsules[bke]=',
        #     tmp_high_capsules[:,:,:5], tmp_seq_lens, tmp_num_high, 
        #     'py_min_max=', py_min_max, 'py_msg=', py_msgs, 'py_seq_diff=', py_seq_diff,
        #     'user_ids=', tmp_debug_feas, 'py_simi=', py_simi, 'py_simi_squash=',
        #     py_simi_squash, tf.shape(tmp_high_capsules)],
        #     message='routing_logits', summarize=100)
        break
      # batch_size x max_k x high_dim(bhe)
      high_capsules = tf.nn.l2_normalize(high_capsules, -1)
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
    return high_capsules, num_high_capsules, tmp_ids
