# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.utils import estimator_utils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
gfile = tf.gfile


class UtilTest(tf.test.TestCase):

  def test_get_ckpt_version(self):
    ver = estimator_utils.get_ckpt_version(
        'oss://easyrec/ckpts/model.ckpt-6500.meta')
    assert ver == 6500, 'invalid version: %s' % str(ver)
    ver = estimator_utils.get_ckpt_version(
        'oss://easyrec/ckpts/model.ckpt-6500')
    assert ver == 6500, 'invalid version: %s' % str(ver)


if __name__ == '__main__':
  tf.test.main()
