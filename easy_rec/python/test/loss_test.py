# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.loss.circle_loss import circle_loss
from easy_rec.python.loss.circle_loss import get_anchor_positive_triplet_mask

from easy_rec.python.loss.f1_reweight_loss import f1_reweight_sigmoid_cross_entropy  # NOQA

from easy_rec.python.loss.softmax_loss_with_negative_mining import softmax_loss_with_negative_mining  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class LossTest(tf.test.TestCase):

  def test_f1_reweighted_loss(self):
    print('test_f1_reweighted_loss')
    logits = tf.constant([0.1, 0.5, 0.3, 0.8, -0.1, 0.3])
    labels = tf.constant([1, 1, 0, 0, 1, 1])
    loss = f1_reweight_sigmoid_cross_entropy(
        labels=labels, logits=logits, beta_square=4)
    with self.test_session() as sess:
      loss_val = sess.run(loss)
      self.assertAlmostEqual(loss_val, 0.47844395, delta=1e-5)

  def test_softmax_loss_with_negative_mining(self):
    print('test_softmax_loss_with_negative_mining')
    user_emb = tf.constant([[0.1, 0.5, 0.3], [0.8, -0.1, 0.3], [0.28, 0.3, 0.9],
                            [0.37, 0.45, 0.93], [-0.7, 0.15, 0.03],
                            [0.18, 0.9, -0.3]])
    item_emb = tf.constant([[0.1, -0.5, 0.3], [0.8, -0.31, 0.3],
                            [0.7, -0.45, 0.15], [0.08, -0.31, -0.9],
                            [-0.7, 0.85, 0.03], [0.18, 0.89, -0.3]])

    label = tf.constant([1, 1, 0, 0, 1, 1])
    loss = softmax_loss_with_negative_mining(
        user_emb, item_emb, label, num_negative_samples=2, seed=1)
    with self.test_session() as sess:
      loss_val = sess.run(loss)
      self.assertAlmostEqual(loss_val, 0.48577175, delta=1e-5)

  def test_circle_loss(self):
    print('test_circle_loss')
    emb = tf.constant([[0.1, 0.2, 0.15, 0.1], [0.3, 0.6, 0.45, 0.3],
                       [0.13, 0.6, 0.45, 0.3], [0.3, 0.26, 0.45, 0.3],
                       [0.3, 0.6, 0.5, 0.13], [0.08, 0.43, 0.21, 0.6]],
                      dtype=tf.float32)
    label = tf.constant([1, 1, 2, 2, 3, 3])
    loss = circle_loss(emb, label, label, margin=0.25, gamma=64)
    with self.test_session() as sess:
      loss_val = sess.run(loss)
      self.assertAlmostEqual(loss_val, 52.75707, delta=1e-5)

  def test_triplet_mask(self):
    print('test_triplet_mask')
    label = tf.constant([1, 1, 2, 2, 3, 3, 4, 5])
    positive_mask = tf.constant(
        [[0., 1., 0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.]],
        dtype=tf.float32)
    negative_mask = tf.constant(
        [[0., 0., 1., 1., 1., 1., 1., 1.], [0., 0., 1., 1., 1., 1., 1., 1.],
         [1., 1., 0., 0., 1., 1., 1., 1.], [1., 1., 0., 0., 1., 1., 1., 1.],
         [1., 1., 1., 1., 0., 0., 1., 1.], [1., 1., 1., 1., 0., 0., 1., 1.],
         [1., 1., 1., 1., 1., 1., 0., 1.], [1., 1., 1., 1., 1., 1., 1., 0.]],
        dtype=tf.float32)
    with self.test_session():
      pos_mask = get_anchor_positive_triplet_mask(label, label)
      self.assertAllEqual(positive_mask, pos_mask)

      neg_mask = _get_anchor_negative_triplet_mask(label, label)
      self.assertAllEqual(negative_mask, neg_mask)

      batch_size = label.shape.as_list()[0]
      neg_mask2 = 1 - pos_mask - tf.eye(batch_size)
      self.assertAllEqual(neg_mask, neg_mask2)


def _get_anchor_negative_triplet_mask(labels, sessions):
  """Return a 2D mask where mask[a, n] is 1.0 iff a and n have distinct session or label.

  Args:
    sessions: a `Tensor` with shape [batch_size]
    labels: a `Tensor` with shape [batch_size]

  Returns:
    mask: tf.bool `Tensor` with shape [batch_size, batch_size]
  """
  # Check if sessions[i] != sessions[k]
  # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
  session_not_equal = tf.not_equal(
      tf.expand_dims(sessions, 0), tf.expand_dims(sessions, 1))

  if labels is sessions:
    return tf.cast(session_not_equal, tf.float32)

  # Check if labels[i] != labels[k]
  # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
  label_not_equal = tf.not_equal(
      tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

  mask = tf.logical_or(session_not_equal, label_not_equal)
  return tf.cast(mask, tf.float32)


if __name__ == '__main__':
  tf.test.main()
