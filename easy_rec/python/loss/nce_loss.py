# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.utils.shape_utils import get_shape_list


def mask_samples(batch_size):
  part = tf.ones((batch_size, batch_size), bool)
  diag_part = tf.linalg.diag_part(part)
  diag_part = tf.fill(tf.shape(diag_part), False)
  part = tf.linalg.set_diag(part, diag_part)
  part_half = tf.concat([part, part], axis=1)
  part_total = tf.concat([part_half, part_half], axis=0)
  return part_total


def nce_loss(z_i, z_j, temp=1):
  batch_size = get_shape_list(z_i)[0]
  N = 2 * batch_size
  z = tf.concat((z_i, z_j), axis=0)
  sim = tf.matmul(z, tf.transpose(z)) / temp
  sim_i_j = tf.matrix_diag_part(
      tf.slice(sim, [batch_size, 0], [batch_size, batch_size]))
  sim_j_i = tf.matrix_diag_part(
      tf.slice(sim, [0, batch_size], [batch_size, batch_size]))
  positive_samples = tf.reshape(tf.concat((sim_i_j, sim_j_i), axis=0), (N, 1))
  mask = mask_samples(batch_size)
  negative_samples = tf.reshape(tf.boolean_mask(sim, mask), (N, -1))

  labels = tf.zeros(N, dtype=tf.int32)
  logits = tf.concat((positive_samples, negative_samples), axis=1)

  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))

  return loss
