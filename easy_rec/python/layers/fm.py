# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class FM:

  def __init__(self, name='fm'):
    """Initializes a `FM` Layer.

    Args:
      name: scope of the FM
    """
    self._name = name

  def __call__(self, fm_fea):
    with tf.name_scope(self._name):
      fm_feas = tf.stack(fm_fea, axis=1)
      sum_square = tf.square(tf.reduce_sum(fm_feas, 1))
      square_sum = tf.reduce_sum(tf.square(fm_feas), 1)
      y_v = 0.5 * tf.subtract(sum_square, square_sum)
    return y_v
