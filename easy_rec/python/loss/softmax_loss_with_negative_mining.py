# coding=utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def support_vector_guided_softmax_loss(pos_score,
                                       neg_scores,
                                       margin=0,
                                       t=1,
                                       smooth=1.0,
                                       threshold=0,
                                       weights=1.0):
  """Refer paper: Support Vector Guided Softmax Loss for Face Recognition (https://128.84.21.199/abs/1812.11317)."""
  new_pos_score = pos_score - margin
  cond = tf.greater_equal(new_pos_score - neg_scores, threshold)
  mask = tf.where(cond, tf.zeros_like(cond, tf.float32),
                  tf.ones_like(cond, tf.float32))  # I_k
  new_neg_scores = mask * (neg_scores * t + t - 1) + (1 - mask) * neg_scores
  logits = tf.concat([new_pos_score, new_neg_scores], axis=1)
  if 1.0 != smooth:
    logits *= smooth

  loss = tf.losses.sparse_softmax_cross_entropy(
      tf.zeros_like(pos_score, dtype=tf.int32), logits, weights=weights)
  # set rank loss to zero if a batch has no positive sample.
  loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
  return loss


def softmax_loss_with_negative_mining(user_emb,
                                      item_emb,
                                      labels,
                                      num_negative_samples=4,
                                      embed_normed=False,
                                      weights=1.0,
                                      gamma=1.0,
                                      margin=0,
                                      t=1,
                                      seed=None):
  """Compute the softmax loss based on the cosine distance explained below.

  Given mini batches for `user_emb` and `item_emb`, this function computes for each element in `user_emb`
  the cosine distance between it and the corresponding `item_emb`,
  and additionally the cosine distance between `user_emb` and some other elements of `item_emb`
   (referred to a negative samples).
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
    gamma: smooth coefficient of softmax
    margin: the margin between positive pair and negative pair
    t: coefficient of support vector guided softmax loss
    seed: A Python integer. Used to create a random seed for the distribution.
      See `tf.set_random_seed`
      for behavior.

  Return:
    support vector guided softmax loss of positive labels
  """
  assert 0 < num_negative_samples, '`num_negative_samples` should be greater than 0'

  batch_size = tf.shape(item_emb)[0]
  is_valid = tf.assert_less(
      num_negative_samples,
      batch_size,
      message='`num_negative_samples` should be less than batch_size')
  with tf.control_dependencies([is_valid]):
    if not embed_normed:
      user_emb = tf.nn.l2_normalize(user_emb, axis=-1)
      item_emb = tf.nn.l2_normalize(item_emb, axis=-1)

    vectors = [item_emb]
    for i in range(num_negative_samples):
      shift = tf.random_uniform([], 1, batch_size, dtype=tf.int32, seed=seed)
      neg_item_emb = tf.roll(item_emb, shift, axis=0)
      vectors.append(neg_item_emb)
    # all_embeddings's shape: (batch_size, num_negative_samples + 1, vec_dim)
    all_embeddings = tf.stack(vectors, axis=1)

    mask = tf.greater(labels, 0)
    mask_user_emb = tf.boolean_mask(user_emb, mask)
    mask_item_emb = tf.boolean_mask(all_embeddings, mask)
    if isinstance(weights, tf.Tensor):
      weights = tf.boolean_mask(weights, mask)

    # sim_scores's shape: (num_of_pos_label_in_batch_size, num_negative_samples + 1)
    sim_scores = tf.keras.backend.batch_dot(
        mask_user_emb, mask_item_emb, axes=(1, 2))
    pos_score = tf.slice(sim_scores, [0, 0], [-1, 1])
    neg_scores = tf.slice(sim_scores, [0, 1], [-1, -1])

    loss = support_vector_guided_softmax_loss(
        pos_score,
        neg_scores,
        margin=margin,
        t=t,
        smooth=gamma,
        weights=weights)
  return loss
