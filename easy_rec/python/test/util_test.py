# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils.dag import DAG
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

  def test_dag(self):
    dag = DAG()
    dag.add_node('a')
    dag.add_node('b')
    dag.add_node('c')
    dag.add_node('d')
    dag.add_edge('a', 'b')
    dag.add_edge('a', 'd')
    dag.add_edge('b', 'c')
    order = dag.topological_sort()
    idx_a = order.index('a')
    idx_b = order.index('b')
    idx_c = order.index('c')
    idx_d = order.index('d')
    assert idx_a < idx_b
    assert idx_a < idx_d
    assert idx_b < idx_c
    c = dag.all_downstreams('b')
    assert c == ['c']
    leaf = dag.all_leaves()
    assert leaf == ['c', 'd']


if __name__ == '__main__':
  tf.test.main()
