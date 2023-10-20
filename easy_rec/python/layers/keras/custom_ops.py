# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for using custom ops."""
import logging
import os
import tensorflow as tf
from tensorflow.python.framework import ops


curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
ops_idr = os.path.dirname(parent_dir)
ops_dir = os.path.join(ops_idr, 'python', 'ops')
if 'PAI' in tf.__version__:
  ops_dir = os.path.join(ops_dir, '1.12_pai')
elif tf.__version__.startswith('1.12'):
  ops_dir = os.path.join(ops_dir, '1.12')
elif tf.__version__.startswith('1.15'):
  if 'IS_ON_PAI' in os.environ:
    ops_dir = os.path.join(ops_dir, 'DeepRec')
  else:
    ops_dir = os.path.join(ops_dir, '1.15')
else:
  ops_dir = None

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class EditDistance(tf.keras.layers.Layer):

  def __init__(self, params, name='edit_distance', reuse=None, **kwargs):
    super(EditDistance, self).__init__(name, **kwargs)
    logging.info("ops_dir is %s" % ops_dir)
    custom_op_path = os.path.join(ops_dir, 'libedit_distance.so')
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
