# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for using custom ops."""
import logging
import os

import tensorflow as tf
from tensorflow.python.framework import ops

import easy_rec

# LIB_PATH = tf.sysconfig.get_link_flags()[0][2:]
# LD_LIBRARY_PATH = os.getenv('LD_LIBRARY_PATH')
# if LIB_PATH not in LD_LIBRARY_PATH:
#   os.environ['LD_LIBRARY_PATH'] = ':'.join([LIB_PATH, LD_LIBRARY_PATH])
#   logging.info('set LD_LIBRARY_PATH=%s' % os.getenv('LD_LIBRARY_PATH'))

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


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
    self.edit_distance = custom_ops.my_edit_distance

    self.txt_encoding = params.get_or_default('text_encoding', 'utf-8')
    self.emb_size = params.get_or_default('embedding_size', 512)
    emb_dim = params.get_or_default('embedding_dim', 4)
    with tf.variable_scope(self.name, reuse=reuse):
      self.embedding_table = tf.get_variable('embedding_table',
                                             [self.emb_size, emb_dim],
                                             tf.float32)

  def call(self, inputs, training=None, **kwargs):
    input1, input2 = inputs[:2]
    with ops.device('/CPU:0'):
      dist = self.edit_distance(
          input1,
          input2,
          normalize=False,
          dtype=tf.int32,
          encoding=self.txt_encoding)
    ids = tf.clip_by_value(dist, 0, self.emb_size - 1)
    embed = tf.nn.embedding_lookup(self.embedding_table, ids)
    return embed
