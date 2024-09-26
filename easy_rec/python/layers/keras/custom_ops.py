# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convenience blocks for using custom ops."""
import logging
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers import Layer

curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
ops_idr = os.path.dirname(parent_dir)
ops_dir = os.path.join(ops_idr, 'ops')
if 'PAI' in tf.__version__:
  ops_dir = os.path.join(ops_dir, '1.12_pai')
elif tf.__version__.startswith('1.12'):
  ops_dir = os.path.join(ops_dir, '1.12')
elif tf.__version__.startswith('1.15'):
  if 'IS_ON_PAI' in os.environ:
    ops_dir = os.path.join(ops_dir, 'DeepRec')
  else:
    ops_dir = os.path.join(ops_dir, '1.15')
elif tf.__version__.startswith('2.12'):
  ops_dir = os.path.join(ops_dir, '2.12')

logging.info('ops_dir is %s' % ops_dir)
custom_op_path = os.path.join(ops_dir, 'libcustom_ops.so')
try:
  custom_ops = tf.load_op_library(custom_op_path)
  logging.info('load custom op from %s succeed' % custom_op_path)
except Exception as ex:
  logging.warning('load custom op from %s failed: %s' %
                  (custom_op_path, str(ex)))
  custom_ops = None

# if tf.__version__ >= '2.0':
#   tf = tf.compat.v1


class SeqAugmentOps(Layer):
  """Do data augmentation for input sequence embedding."""

  def __init__(self, params, name='sequence_aug', reuse=None, **kwargs):
    super(SeqAugmentOps, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self.seq_aug_params = params.get_pb_config()
    self.seq_augment = custom_ops.my_seq_augment

  def call(self, inputs, training=None, **kwargs):
    assert isinstance(
        inputs,
        (list, tuple)), 'the inputs of SeqAugmentOps must be type of list/tuple'
    assert len(inputs) >= 2, 'SeqAugmentOps must have at least 2 inputs'
    seq_input, seq_len = inputs[:2]
    embedding_dim = int(seq_input.shape[-1])
    with tf.variable_scope(self.name, reuse=self.reuse):
      mask_emb = tf.get_variable(
          'mask', (embedding_dim,), dtype=tf.float32, trainable=True)
    seq_len = tf.to_int32(seq_len)
    with ops.device('/CPU:0'):
      aug_seq, aug_len = self.seq_augment(seq_input, seq_len, mask_emb,
                                          self.seq_aug_params.crop_rate,
                                          self.seq_aug_params.reorder_rate,
                                          self.seq_aug_params.mask_rate)
    return aug_seq, aug_len


class TextNormalize(Layer):

  def __init__(self, params, name='text_normalize', reuse=None, **kwargs):
    super(TextNormalize, self).__init__(name=name, **kwargs)
    self.txt_normalizer = custom_ops.text_normalize_op
    self.norm_parameter = params.get_or_default('norm_parameter', 0)
    self.remove_space = params.get_or_default('remove_space', False)

  def call(self, inputs, training=None, **kwargs):
    inputs = inputs if type(inputs) in (tuple, list) else [inputs]
    with ops.device('/CPU:0'):
      result = [
          self.txt_normalizer(
              txt,
              parameter=self.norm_parameter,
              remove_space=self.remove_space) for txt in inputs
      ]
    if len(result) == 1:
      return result[0]
    return result


class MappedDotProduct(Layer):

  def __init__(self, params, name='mapped_dot_product', reuse=None, **kwargs):
    super(MappedDotProduct, self).__init__(name=name, **kwargs)
    self.mapped_dot_product = custom_ops.mapped_dot_product
    self.bucketize = custom_ops.my_bucketize
    self.default_value = params.get_or_default('default_value', 0)
    self.separator = params.get_or_default('separator', '\035')
    self.norm_fn = params.get_or_default('normalize_fn', None)
    self.boundaries = list(params.get_or_default('boundaries', []))
    self.emb_dim = params.get_or_default('embedding_dim', 0)
    self.print_first_n = params.get_or_default('print_first_n', 0)
    self.summarize = params.get_or_default('summarize', None)
    if self.emb_dim > 0:
      vocab_size = len(self.boundaries) + 1
      with tf.variable_scope(self.name, reuse=reuse):
        self.embedding_table = tf.get_variable(
            name='dot_product_emb_table',
            shape=[vocab_size, self.emb_dim],
            dtype=tf.float32)

  def call(self, inputs, training=None, **kwargs):
    query, doc = inputs[:2]
    with ops.device('/CPU:0'):
      feature = self.mapped_dot_product(
          query=query,
          document=doc,
          feature_name=self.name,
          separator=self.separator,
          default_value=self.default_value)
      tf.summary.scalar(self.name, tf.reduce_mean(feature))
      if self.print_first_n:
        encode_q = tf.regex_replace(query, self.separator, ' ')
        encode_t = tf.regex_replace(query, self.separator, ' ')
        feature = tf.Print(
            feature, [encode_q, encode_t, feature],
            message=self.name,
            first_n=self.print_first_n,
            summarize=self.summarize)
      if self.norm_fn is not None:
        fn = eval(self.norm_fn)
        feature = fn(feature)
        tf.summary.scalar('normalized_%s' % self.name, tf.reduce_mean(feature))
        if self.print_first_n:
          feature = tf.Print(
              feature, [feature],
              message='normalized %s' % self.name,
              first_n=self.print_first_n,
              summarize=self.summarize)
      if self.boundaries:
        feature = self.bucketize(feature, boundaries=self.boundaries)
        tf.summary.histogram('bucketized_%s' % self.name, feature)
    if self.emb_dim > 0 and self.boundaries:
      vocab_size = len(self.boundaries) + 1
      one_hot_input_ids = tf.one_hot(feature, depth=vocab_size)
      return tf.matmul(one_hot_input_ids, self.embedding_table)
    return tf.expand_dims(feature, axis=-1)


class OverlapFeature(Layer):

  def __init__(self, params, name='overlap_feature', reuse=None, **kwargs):
    super(OverlapFeature, self).__init__(name=name, **kwargs)
    self.overlap_feature = custom_ops.overlap_fg_op
    methods = params.get_or_default('methods', [])
    assert methods, 'overlap feature methods must be set'
    self.methods = [str(method) for method in methods]
    self.norm_fn = params.get_or_default('normalize_fn', None)
    self.boundaries = list(params.get_or_default('boundaries', []))
    self.separator = params.get_or_default('separator', '\035')
    self.default_value = params.get_or_default('default_value', '-1')
    self.emb_dim = params.get_or_default('embedding_dim', 0)
    self.print_first_n = params.get_or_default('print_first_n', 0)
    self.summarize = params.get_or_default('summarize', None)
    if self.emb_dim > 0:
      vocab_size = len(self.boundaries) + 1
      vocab_size *= len(self.methods)
      with tf.variable_scope(self.name, reuse=reuse):
        self.embedding_table = tf.get_variable(
            name='overlap_emb_table',
            shape=[vocab_size, self.emb_dim],
            dtype=tf.float32)

  def call(self, inputs, training=None, **kwargs):
    query, title = inputs[:2]
    with ops.device('/CPU:0'):
      feature = self.overlap_feature(
          query=query,
          title=title,
          feature_name=self.name,
          separator=self.separator,
          default_value=self.default_value,
          boundaries=self.boundaries,
          methods=self.methods,
          dtype=tf.int32 if self.boundaries else tf.float32)

    for i, method in enumerate(self.methods):
      # warning: feature[:, i] may be not the result of method
      if self.boundaries:
        tf.summary.histogram('bucketized_%s' % method, feature[:, i])
      else:
        tf.summary.scalar(method, tf.reduce_mean(feature[:, i]))
    if self.print_first_n:
      encode_q = tf.regex_replace(query, self.separator, ' ')
      encode_t = tf.regex_replace(query, self.separator, ' ')
      feature = tf.Print(
          feature, [encode_q, encode_t, feature],
          message=self.name,
          first_n=self.print_first_n,
          summarize=self.summarize)
    if self.norm_fn is not None:
      fn = eval(self.norm_fn)
      feature = fn(feature)

    if self.emb_dim > 0 and self.boundaries:
      # This vocab will be small so we always do one-hot here, since it is always
      # faster for a small vocabulary.
      batch_size = tf.shape(feature)[0]
      vocab_size = len(self.boundaries) + 1
      num_indices = len(self.methods)
      # Compute offsets, add to every column indices
      offsets = tf.range(num_indices) * vocab_size  # Shape: [3]
      offsets = tf.reshape(offsets, [1, num_indices])  # Shape: [1, 3]
      offsets = tf.tile(offsets,
                        [batch_size, 1])  # Shape: [batch_size, num_indices]
      shifted_indices = feature + offsets  # Shape: [batch_size, num_indices]
      flat_feature_ids = tf.reshape(shifted_indices, [-1])
      one_hot_ids = tf.one_hot(flat_feature_ids, depth=vocab_size * num_indices)
      feature_embeddings = tf.matmul(one_hot_ids, self.embedding_table)
      feature_embeddings = tf.reshape(feature_embeddings,
                                      [batch_size, num_indices * self.emb_dim])
      return feature_embeddings
    return feature


class EditDistance(Layer):

  def __init__(self, params, name='edit_distance', reuse=None, **kwargs):
    super(EditDistance, self).__init__(name=name, **kwargs)
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
