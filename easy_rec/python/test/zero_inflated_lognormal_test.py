# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import tensorflow as tf
from scipy import stats

from easy_rec.python.loss.zero_inflated_lognormal import zero_inflated_lognormal_loss  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

# Absolute error tolerance in asserting array near.
_ERR_TOL = 1e-6


# softplus function that calculates log(1+exp(x))
def _softplus(x):
  return np.log(1.0 + np.exp(x))


# sigmoid function that calculates 1/(1+exp(-x))
def _sigmoid(x):
  return 1 / (1 + np.exp(-x))


class ZeroInflatedLognormalLossTest(tf.test.TestCase):

  def setUp(self):
    super(ZeroInflatedLognormalLossTest, self).setUp()
    self.logits = np.array([[.1, .2, .3], [.4, .5, .6]])
    self.labels = np.array([[0.], [1.5]])

  def zero_inflated_lognormal(self, labels, logits):
    positive_logits = logits[..., :1]
    loss_zero = _softplus(positive_logits)
    loc = logits[..., 1:2]
    scale = np.maximum(
        _softplus(logits[..., 2:]), np.sqrt(tf.keras.backend.epsilon()))
    log_prob_non_zero = stats.lognorm.logpdf(
        x=labels, s=scale, loc=0, scale=np.exp(loc))
    loss_non_zero = _softplus(-positive_logits) - log_prob_non_zero
    return np.mean(np.where(labels == 0., loss_zero, loss_non_zero), axis=-1)

  def test_loss_value(self):
    expected_loss = self.zero_inflated_lognormal(self.labels, self.logits)
    expected_loss = np.average(expected_loss)
    loss = zero_inflated_lognormal_loss(self.labels, self.logits)
    self.assertNear(self.evaluate(loss), expected_loss, _ERR_TOL)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
