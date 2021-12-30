# coding=utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def circle_loss(embeddings,
                labels,
                sessions=None,
                margin=0.25,
                gamma=32,
                embed_normed=False):
  """Paper: Circle Loss: A Unified Perspective of Pair Similarity Optimization.

  Link: http://arxiv.org/pdf/2002.10857.pdf

  Args:
    embeddings: A `Tensor` with shape [batch_size, embedding_size]. The embedding of each sample.
    labels: a `Tensor` with shape [batch_size]. e.g. click or not click in the session.
    sessions: a `Tensor` with shape [batch_size]. session ids of each sample.
    margin: the margin between positive similarity and negative similarity
    gamma: parameter of circle loss
    embed_normed: bool, whether input embeddings l2 normalized
  """
  norm_embeddings = embeddings if embed_normed else tf.nn.l2_normalize(
      embeddings, axis=-1)
  pair_wise_cosine_matrix = tf.matmul(
      norm_embeddings, norm_embeddings, transpose_b=True)

  positive_mask = get_anchor_positive_triplet_mask(labels, sessions)
  negative_mask = 1 - positive_mask - tf.eye(tf.shape(labels)[0])

  delta_p = 1 - margin
  delta_n = margin

  ap = tf.nn.relu(-tf.stop_gradient(pair_wise_cosine_matrix * positive_mask) +
                  1 + margin)
  an = tf.nn.relu(
      tf.stop_gradient(pair_wise_cosine_matrix * negative_mask) + margin)

  logit_p = -ap * (pair_wise_cosine_matrix -
                   delta_p) * gamma * positive_mask - (1 - positive_mask) * 1e12
  logit_n = an * (pair_wise_cosine_matrix -
                  delta_n) * gamma * negative_mask - (1 - negative_mask) * 1e12

  joint_neg_loss = tf.reduce_logsumexp(logit_n, axis=-1)
  joint_pos_loss = tf.reduce_logsumexp(logit_p, axis=-1)
  loss = tf.nn.softplus(joint_neg_loss + joint_pos_loss)
  return tf.reduce_mean(loss)


def get_anchor_positive_triplet_mask(labels, sessions=None):
  """Return a 2D mask where mask[a, p] is 1.0 iff a and p are distinct and have same session and label.

  Args:
    labels: a `Tensor` with shape [batch_size]
    sessions: a `Tensor` with shape [batch_size]

  Returns:
    mask: tf.float32 `Tensor` with shape [batch_size, batch_size]
  """
  # Check that i and j are distinct
  indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
  indices_not_equal = tf.logical_not(indices_equal)

  # Check if labels[i] == labels[j]
  # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
  labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

  # Check if labels[i] == labels[j]
  if sessions is None or sessions is labels:
    class_equal = labels_equal
  else:
    sessions_equal = tf.equal(
        tf.expand_dims(sessions, 0), tf.expand_dims(sessions, 1))
    class_equal = tf.logical_and(sessions_equal, labels_equal)

  # Combine the three masks
  mask = tf.logical_and(indices_not_equal, class_equal)
  return tf.cast(mask, tf.float32)
