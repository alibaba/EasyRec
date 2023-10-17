# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for using custom ops."""
import logging
import os

import tensorflow as tf

import easy_rec

LIB_PATH = tf.sysconfig.get_link_flags()[0][2:]
LD_LIBRARY_PATH = os.getenv('LD_LIBRARY_PATH')
os.environ['LD_LIBRARY_PATH'] = ':'.join([LIB_PATH, LD_LIBRARY_PATH])
logging.info('set LD_LIBRARY_PATH=%s' % os.getenv('LD_LIBRARY_PATH'))


class EditDistance(tf.keras.layers.Layer):

  def __init__(self, params, name='edit_distance', reuse=None, **kwargs):
    super(EditDistance, self).__init__(name, **kwargs)

    custom_op_path = os.path.join(easy_rec.ops_dir, 'libedit_distance.so')
    try:
      custom_ops = tf.load_op_library(custom_op_path)
      logging.info('load edit_distance op from %s succeed' % custom_op_path)
    except Exception as ex:
      logging.warning('load edit_distance op from %s failed: %s' %
                      (custom_op_path, str(ex)))
      custom_ops = None
    self.edit_distance = custom_ops.edit_distance_op

  def call(self, inputs, training=None, **kwargs):
    input1, input2 = inputs[:2]
    print('input1:', input1)
    print('input2:', input2)
    str1 = tf.sparse.to_dense(input1, default_value='')
    str2 = tf.sparse.to_dense(input1, default_value='')
    print('str1:', str1)
    print('str2:', str2)
    dist = self.edit_distance(str1, str2, dtype=tf.float32)
    print('dist:', dist)
    dist = tf.reshape(dist, [-1, 1])
    return dist
