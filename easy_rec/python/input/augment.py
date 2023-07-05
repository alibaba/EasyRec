# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def assign(input_tensor, position=None, value=None):
  input_tensor[tuple(position)] = value
  return input_tensor


def item_mask(aug_data, length, gamma=0.3):
  length1 = tf.cast(length, dtype=tf.float32)
  num_mask = tf.cast(tf.math.floor(length1 * gamma), dtype=tf.int32)
  seq = tf.range(length, dtype=tf.int32)
  mask_index = tf.random.shuffle(seq)[:num_mask]
  masked_item_seq = aug_data
  masked_item_seq = tf.py_func(
      assign,
      inp=[masked_item_seq, [mask_index], 0],
      Tout=masked_item_seq.dtype)
  return masked_item_seq, length


def item_crop(aug_data, length, eta=0.6):
  length1 = tf.cast(length, dtype=tf.float32)
  max_length = tf.cast(get_shape_list(aug_data)[0], dtype=tf.int32)
  embedding_size = get_shape_list(aug_data)[1]

  num_left = tf.cast(tf.math.floor(length1 * eta), dtype=tf.int32)
  crop_begin = tf.random.uniform([1],
                                 minval=0,
                                 maxval=length - num_left,
                                 dtype=tf.int32)[0]
  cropped_item_seq = tf.zeros([get_shape_list(aug_data)[0], embedding_size])
  cropped_item_seq = tf.where(
      crop_begin + num_left < max_length,
      tf.concat([
          aug_data[crop_begin:crop_begin + num_left],
          cropped_item_seq[:max_length - num_left]
      ],
                axis=0),
      tf.concat([aug_data[crop_begin:], cropped_item_seq[:crop_begin]], axis=0))
  return cropped_item_seq, num_left


def item_reorder(aug_data, length, beta=0.6):
  length1 = tf.cast(length, dtype=tf.float32)
  num_reorder = tf.cast(tf.math.floor(length1 * beta), dtype=tf.int32)
  reorder_begin = tf.random.uniform([1],
                                    minval=0,
                                    maxval=length - num_reorder,
                                    dtype=tf.int32)[0]
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


def augment(x):
  seq, length = x
  flag = tf.range(3, dtype=tf.int32)
  flag1 = tf.random.shuffle(flag)[:1][0]
  aug_seq, aug_len = tf.cond(
      tf.equal(flag1, 0), lambda: item_crop(seq, length), lambda: tf.cond(
          tf.equal(flag1, 1), lambda: item_mask(seq, length), lambda:
          item_reorder(seq, length)))

  return [aug_seq, aug_len]


def input_aug_data(original_data, seq_len):
  print('seq_len:', seq_len)
  lengths = tf.cast(seq_len, dtype=tf.int32)
  aug_seq1, aug_len1 = tf.map_fn(
      augment, elems=(original_data, lengths), dtype=[tf.float32, tf.int32])
  aug_seq2, aug_len2 = tf.map_fn(
      augment, elems=(original_data, lengths), dtype=[tf.float32, tf.int32])
  aug_seq1 = tf.reshape(aug_seq1, tf.shape(original_data))
  aug_seq2 = tf.reshape(aug_seq2, tf.shape(original_data))
  return aug_seq1, aug_seq2, aug_len1, aug_len2
