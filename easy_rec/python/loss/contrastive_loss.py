# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def l2_loss(x1, x2):
  loss = tf.pow(tf.norm(x1 - x2, axis=1), 2)
  return tf.reduce_mean(loss)


def info_nce_loss(query, positive, temperature=0.1):
  """Calculates the InfoNCE loss for self-supervised learning.

  This contrastive loss enforces the embeddings of similar (positive) samples to be close
      and those of different (negative) samples to be distant.
  A query embedding is compared with one positive key and with one or more negative keys.

  References:
      https://arxiv.org/abs/1807.03748v2
      https://arxiv.org/abs/2010.05113
  """
  # Check input dimensionality.
  if query.shape.ndims != 2:
    raise ValueError('<query> must have 2 dimensions.')
  if positive.shape.ndims != 2:
    raise ValueError('<positive> must have 2 dimensions.')
  # Embedding vectors should have same number of components.
  if query.shape[-1] != positive.shape[-1]:
    raise ValueError(
        'Vectors of <query> and <positive> should have the same number of components.'
    )

  # Negative keys are implicitly off-diagonal positive keys.

  # Cosine between all combinations
  logits = tf.matmul(query, positive, transpose_b=True)
  logits /= temperature

  # Positive keys are the entries on the diagonal
  batch_size = tf.shape(query)[0]
  labels = tf.range(batch_size)

  return tf.losses.sparse_softmax_cross_entropy(labels, logits)
