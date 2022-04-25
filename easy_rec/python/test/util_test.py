# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils.expr_util import get_expression

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

  def test_get_expression_greater(self):
    result = get_expression('age_level>item_age_level',
                            ['age_level', 'item_age_level'])
    assert result == "tf.greater(parsed_dict['age_level'], parsed_dict['item_age_level'])"

  def test_get_expression_greater_equal(self):
    result = get_expression('age_level>=item_age_level',
                            ['age_level', 'item_age_level'])
    assert result == "tf.greater_equal(parsed_dict['age_level'], parsed_dict['item_age_level'])"

  def test_get_expression_less(self):
    result = get_expression('age_level<item_age_level',
                            ['age_level', 'item_age_level'])
    assert result == "tf.less(parsed_dict['age_level'], parsed_dict['item_age_level'])"

  def test_get_expression_less_equal(self):
    result = get_expression('age_level<=item_age_level',
                            ['age_level', 'item_age_level'])
    assert result == "tf.less_equal(parsed_dict['age_level'], parsed_dict['item_age_level'])"

  def test_get_expression_and(self):
    result = get_expression('(age_level>3)&(item_age_level<1)',
                            ['age_level', 'item_age_level'])
    assert result == "tf.greater(parsed_dict['age_level'], 3) & tf.less(parsed_dict['item_age_level'], 1)"

    result = get_expression(
        '(age_level>item_age_level) & (age_level<item_age_level*3)',
        ['age_level', 'item_age_level'])
    assert result == "tf.greater(parsed_dict['age_level'], parsed_dict['item_age_level']) &" \
                     " tf.less(parsed_dict['age_level'], parsed_dict['item_age_level']*3)"

  def test_get_expression_or(self):
    result = get_expression('(age_level>3)|(item_age_level<1)',
                            ['age_level', 'item_age_level'])
    assert result == "tf.greater(parsed_dict['age_level'], 3) | tf.less(parsed_dict['item_age_level'], 1)"


if __name__ == '__main__':
  tf.test.main()
