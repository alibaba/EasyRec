# coding=utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def softmax_loss_with_negative_mining(user_emb,
                                      item_emb,
                                      labels,
                                      num_negative_samples=4,
                                      embed_normed=False,
                                      weights=1.0,
                                      gamma=1.0):
  """Given mini batches for `user_emb` and `item_emb`, this function computes for each element in `user_emb` the cosine distance between it and the corresponding `item_emb` and additionally the cosine distance between `user_emb` and some other elements of `item_emb` (referred to a negative samples).

  The negative samples are formed on the fly by shifting the right side (`item_emb`).
    Then the softmax loss will be computed based on these cosine distance.
  Args:
    user_emb: A `Tensor` with shape [batch_size, embedding_size]. The embedding of user.
    item_emb: A `Tensor` with shape [batch_size, embedding_size]. The embedding of item.
    labels: a `Tensor` with shape [batch_size]. e.g. click or not click in the session. It's values must be 0 or 1.
    num_negative_samples: the num of negative samples, should be in range [1, batch_size).
    embed_normed: bool, whether input embeddings l2 normalized
    weights: `weights` acts as a coefficient for the loss. If a scalar is provided,
      then the loss is simply scaled by the given value. If `weights` is a
      tensor of shape `[batch_size]`, then the loss weights apply to each corresponding sample.
    gamma: parameter of softmax
  Return:
    A tuple of (log loss, the probability of the similarity between anchor and the other samples, similarities)
  """
  if not embed_normed:
    user_emb = tf.nn.l2_normalize(user_emb, axis=-1)
    item_emb = tf.nn.l2_normalize(item_emb, axis=-1)

  batch_size = get_shape_list(item_emb)[0]
  vectors = [item_emb]
  for i in range(num_negative_samples):
    shift = tf.random_uniform([], 1, batch_size, dtype=tf.int32)
    neg_item_emb = tf.roll(item_emb, shift, axis=0)
    vectors.append(neg_item_emb)
  # all_embeddings's shape: (batch_size, num_negative_samples + 1, vec_dim)
  all_embeddings = tf.stack(vectors, axis=1)

  # sim_scores's shape: (batch_size, num_negative_samples + 1)
  sim_scores = tf.keras.backend.batch_dot(user_emb, all_embeddings, axes=(1, 2))
  probs = tf.nn.softmax(sim_scores * gamma)
  # fetch the first column, the probability of positive sample
  pos_prob = tf.squeeze(tf.slice(probs, [0, 0], [-1, 1]))

  labels = tf.to_float(labels)
  loss = tf.losses.compute_weighted_loss(-tf.log(pos_prob) * labels,
                                         weights * labels)
  return loss, probs, sim_scores
