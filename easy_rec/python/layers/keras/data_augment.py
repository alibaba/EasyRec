# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def item_mask(aug_data, length, mask_emb, mask_rate):
  length1 = tf.cast(length, dtype=tf.float32)
  num_mask = tf.cast(tf.math.floor(length1 * mask_rate), dtype=tf.int32)
  max_len = tf.shape(aug_data)[0]
  seq_mask = tf.sequence_mask(num_mask, length)
  seq_mask = tf.random.shuffle(seq_mask)
  padding = tf.sequence_mask(0, max_len - length)
  seq_mask = tf.concat([seq_mask, padding], axis=0)

  mask_emb = tf.tile(mask_emb, [max_len, 1])

  masked_item_seq = tf.where(seq_mask, mask_emb, aug_data)
  return masked_item_seq, length


def item_crop(aug_data, length, crop_rate):
  length1 = tf.cast(length, dtype=tf.float32)
  max_len, _ = get_shape_list(aug_data)
  max_length = tf.cast(max_len, dtype=tf.int32)

  num_left = tf.cast(tf.math.floor(length1 * crop_rate), dtype=tf.int32)
  crop_begin = tf.random.uniform([],
                                 minval=0,
                                 maxval=length - num_left,
                                 dtype=tf.int32)
  zeros = tf.zeros_like(aug_data)
  x = aug_data[crop_begin:crop_begin + num_left]
  y = zeros[:max_length - num_left]
  cropped = tf.concat([x, y], axis=0)
  cropped_item_seq = tf.where(
      crop_begin + num_left < max_length, cropped,
      tf.concat([aug_data[crop_begin:], zeros[:crop_begin]], axis=0))
  return cropped_item_seq, num_left


def item_reorder(aug_data, length, reorder_rate):
  length1 = tf.cast(length, dtype=tf.float32)
  num_reorder = tf.cast(tf.math.floor(length1 * reorder_rate), dtype=tf.int32)
  reorder_begin = tf.random.uniform([],
                                    minval=0,
                                    maxval=length - num_reorder,
                                    dtype=tf.int32)
  shuffle_index = tf.range(reorder_begin, reorder_begin + num_reorder)
  shuffle_index = tf.random.shuffle(shuffle_index)
  x = tf.range(get_shape_list(aug_data)[0])
  left = tf.slice(x, [0], [reorder_begin])
  right = tf.slice(x, [reorder_begin + num_reorder], [-1])
  reordered_item_index = tf.concat([left, shuffle_index, right], axis=0)
  reordered_item_seq = tf.scatter_nd(
      tf.expand_dims(reordered_item_index, axis=1), aug_data,
      tf.shape(aug_data))
  return reordered_item_seq, length


def augment_fn(x, aug_param, mask):
  seq, length = x

  def crop_fn():
    return item_crop(seq, length, aug_param.crop_rate)

  def mask_fn():
    return item_mask(seq, length, mask, aug_param.mask_rate)

  def reorder_fn():
    return item_reorder(seq, length, aug_param.reorder_rate)

  method = tf.random.uniform([], minval=0, maxval=3, dtype=tf.int32)

  aug_seq, aug_len = tf.cond(
      tf.equal(method, 0), crop_fn,
      lambda: tf.cond(tf.equal(method, 1), mask_fn, reorder_fn))

  return aug_seq, aug_len


def sequence_augment(seq_input, seq_len, mask, aug_param):
  lengths = tf.cast(seq_len, dtype=tf.int32)
  aug_seq, aug_len = tf.map_fn(
      lambda elems: augment_fn(elems, aug_param, mask),
      elems=(seq_input, lengths),
      dtype=(tf.float32, tf.int32))

  aug_seq = tf.reshape(aug_seq, tf.shape(seq_input))
  return aug_seq, aug_len


class SeqAugment(Layer):
  """Do data augmentation for input sequence embedding."""

  def __init__(self, params, name='seq_aug', reuse=None, **kwargs):
    super(SeqAugment, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self.seq_aug_params = params.get_pb_config()

  def call(self, inputs, training=None, **kwargs):
    assert isinstance(inputs, (list, tuple))
    seq_input, seq_len = inputs[:2]

    embedding_size = int(seq_input.shape[-1])
    with tf.variable_scope(self.name, reuse=self.reuse):
      mask_emb = tf.get_variable(
          'mask', [1, embedding_size], dtype=tf.float32, trainable=True)

    aug_seq, aug_len = sequence_augment(seq_input, seq_len, mask_emb,
                                        self.seq_aug_params)
    return aug_seq, aug_len
