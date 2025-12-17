# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import tensorflow as tf
from scipy import stats

from easy_rec.python.loss.zero_inflated_lognormal import (  # NOQA
    zero_inflated_lognormal_loss,)

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class ZeroInflatedLognormalLossTest(tf.test.TestCase):

  def setUp(self):
    super(ZeroInflatedLognormalLossTest, self).setUp()
    self.logits = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    self.labels = np.array([[0.0], [1.5]])

  def zero_inflated_lognormal(self, labels, logits, max_sigma=5.0):
    labels = labels.astype(np.float32)
    if labels.ndim == 1:
      labels = labels[:, None]
    logits = logits.astype(np.float32)
    positive = (labels > 0).astype(np.float32)  # [B,1]

    z = logits[..., :1]
    mu = logits[..., 1:2]
    sigma = np.log1p(np.exp(logits[..., 2:]))  # softplus
    sigma = np.clip(sigma, np.sqrt(tf.keras.backend.epsilon()), max_sigma)

    # 分类损失：BCE with logits，按 batch 平均
    # bce(z, y) = max(z,0) - z*y + log(1 + exp(-|z|))
    bce = np.maximum(z, 0.0) - z * positive + np.log1p(np.exp(-np.abs(z)))
    classification_loss = np.mean(bce)

    # 回归损失：仅对正样本的 log_prob 求平均（按正样本数）
    # 避免对 y=0 求 logpdf
    mask = (labels > 0).squeeze(-1)
    logprob = np.zeros_like(labels, dtype=np.float64)
    if np.any(mask):
      # scipy 参数化：s=shape= sigma, scale=exp(mu), loc=0
      logprob[mask, 0] = stats.lognorm.logpdf(
          x=labels[mask, 0].astype(np.float64),
          s=sigma[mask, 0].astype(np.float64),
          loc=0.0,
          scale=np.exp(mu[mask, 0].astype(np.float64)),
      )
    num_pos = np.sum(positive) + 1e-8
    regression_loss = -(np.sum(positive * logprob) / num_pos)
    # 与实现一致的组合（此处测试会把正则项设为0）
    total = classification_loss + regression_loss
    return float(total)

  def test_loss_value(self):
    expected_loss = self.zero_inflated_lognormal(self.labels, self.logits)
    expected_loss = np.average(expected_loss)
    loss = zero_inflated_lognormal_loss(
        self.labels, self.logits, mu_reg=0, sigma_reg=0)
    # Absolute error tolerance in asserting array near.
    _ERR_TOL = 1e-6
    self.assertNear(self.evaluate(loss), expected_loss, _ERR_TOL)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
