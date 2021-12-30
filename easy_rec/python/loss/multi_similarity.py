# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.loss.circle_loss import get_anchor_positive_triplet_mask
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def ms_loss(embeddings,
            labels,
            session_ids=None,
            alpha=2.0,
            beta=50.0,
            lamb=1.0,
            eps=0.1,
            ms_mining=False,
            embed_normed=False):
  """Refer paper: Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning.

  ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/
    Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
  """
  # make sure embedding should be l2-normalized
  if not embed_normed:
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
  labels = tf.reshape(labels, [-1, 1])

  embed_shape = get_shape_list(embeddings)
  batch_size = embed_shape[0]

  mask_pos = get_anchor_positive_triplet_mask(labels, session_ids)
  mask_neg = 1 - mask_pos - tf.eye(batch_size)

  sim_mat = tf.matmul(
      embeddings, embeddings, transpose_a=False, transpose_b=True)
  sim_mat = tf.maximum(sim_mat, 0.0)

  pos_mat = tf.multiply(sim_mat, mask_pos)
  neg_mat = tf.multiply(sim_mat, mask_neg)

  if ms_mining:
    max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
    tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
    min_val = tf.reduce_min(
        tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1,
        keepdims=True) + tmp_max_val

    max_val = tf.tile(max_val, [1, batch_size])
    min_val = tf.tile(min_val, [1, batch_size])

    mask_pos = tf.where(pos_mat < max_val + eps, mask_pos,
                        tf.zeros_like(mask_pos))
    mask_neg = tf.where(neg_mat > min_val - eps, mask_neg,
                        tf.zeros_like(mask_neg))

  pos_exp = tf.exp(-alpha * (pos_mat - lamb))
  pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

  neg_exp = tf.exp(beta * (neg_mat - lamb))
  neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

  pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
  neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

  loss = tf.reduce_mean(pos_term + neg_term)
  return loss
