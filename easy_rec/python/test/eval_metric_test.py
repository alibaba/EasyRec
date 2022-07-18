# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import division

import logging

import tensorflow as tf
from absl.testing import parameterized

from easy_rec.python.utils.test_utils import RunAsSubprocess

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MetricsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  @RunAsSubprocess
  def test_max_f1(self):
    from easy_rec.python.core.metrics import max_f1
    labels = tf.constant([1, 0, 0, 1], dtype=tf.int32)
    probs = tf.constant([0.9, 0.8, 0.7, 0.6], dtype=tf.float32)
    f1, f1_update_op = max_f1(labels, probs)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(f1_update_op)
      f1_score = sess.run(f1)
    self.assertAlmostEqual(f1_score, 2.0 / 3)

  @RunAsSubprocess
  def test_gauc_all_negative_label(self):
    from easy_rec.python.core.metrics import gauc
    labels = tf.constant([0, 0, 0, 0], dtype=tf.int32)
    probs = tf.constant([0.9, 0.8, 0.7, 0.6], dtype=tf.float32)
    uids = tf.constant([1, 1, 1, 1], dtype=tf.int32)
    value_op, update_op = gauc(labels, probs, uids)
    with tf.Session() as sess:
      sess.run(update_op)
      score = sess.run(value_op)
    self.assertAlmostEqual(score, 0.0)

  @parameterized.named_parameters(
      [['_reduction_mean', 'mean', 0.5833333],
       ['_reduction_mean_by_sample_num', 'mean_by_sample_num', 0.5925926],
       ['_reduction_mean_by_positive_num', 'mean_by_positive_num', 0.6]])
  @RunAsSubprocess
  def test_gauc(self, reduction, expected):
    from easy_rec.python.core.metrics import gauc
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))
    probs = tf.placeholder(dtype=tf.float32, shape=(None,))
    uids = tf.placeholder(dtype=tf.int32, shape=(None,))
    value_op, update_op = gauc(labels, probs, uids, reduction=reduction)
    with tf.Session() as sess:
      sess.run(
          update_op,
          feed_dict={
              labels: [1, 0, 1, 1, 0],
              probs: [0.9, 0.8, 0.7, 0.6, 0.5],
              uids: [1, 1, 1, 1, 1]
          })
      sess.run(
          update_op,
          feed_dict={
              labels: [1, 0, 0, 1],
              probs: [0.9, 0.8, 0.7, 0.6],
              uids: [2, 2, 2, 2]
          })
      score = sess.run(value_op)
    self.assertAlmostEqual(score, expected)

  @parameterized.named_parameters(
      [['_reduction_mean', 'mean', 0.5833333],
       ['_reduction_mean_by_sample_num', 'mean_by_sample_num', 0.5925926],
       ['_reduction_mean_by_positive_num', 'mean_by_positive_num', 0.6]])
  @RunAsSubprocess
  def test_session_auc(self, reduction, expected):
    from easy_rec.python.core.metrics import session_auc
    labels = tf.placeholder(dtype=tf.int32, shape=(None,))
    probs = tf.placeholder(dtype=tf.float32, shape=(None,))
    session_ids = tf.placeholder(dtype=tf.int32, shape=(None,))
    value_op, update_op = session_auc(
        labels, probs, session_ids, reduction=reduction)
    with tf.Session() as sess:
      sess.run(
          update_op,
          feed_dict={
              labels: [1, 0, 1, 1, 0],
              probs: [0.9, 0.8, 0.7, 0.6, 0.5],
              session_ids: [1, 1, 1, 1, 1]
          })
      sess.run(
          update_op,
          feed_dict={
              labels: [1, 0, 0, 1],
              probs: [0.9, 0.8, 0.7, 0.6],
              session_ids: [2, 2, 2, 2]
          })
      score = sess.run(value_op)
    self.assertAlmostEqual(score, expected)


if __name__ == '__main__':
  tf.test.main()
