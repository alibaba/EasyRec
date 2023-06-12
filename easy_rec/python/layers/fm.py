# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class FM:

  def __init__(self, name='fm'):
    """Initializes a `FM` Layer.

    Args:
      name: scope of the FM
    """
    self._name = name

  def __call__(self, fm_fea):
    with tf.name_scope(self._name):
      fm_feas = [tf.expand_dims(x, axis=1) for x in fm_fea]
      fm_feas = tf.concat(fm_feas, axis=1)
      sum_square = tf.square(tf.reduce_sum(fm_feas, 1))
      square_sum = tf.reduce_sum(tf.square(fm_feas), 1)
      y_v = 0.5 * tf.subtract(sum_square, square_sum)
    return y_v


class FMLayer(object):
  """Factorization Machine models pairwise (order-2) feature interactions
   without linear term and bias.
    Input shape
      - List of 2D tensor with shape: ``(batch_size,embedding_size)``.
      - Or a 3D tensor with shape: ``(batch_size,field_size,embedding_size)``
    Output shape
      - 2D tensor with shape: ``(batch_size, 1)``.
    References
      - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  """
  def __call__(self, inputs):
    if type(inputs) == list:
      emb_dims = set()
      for x in inputs:
        emb_dims.add(int(x.shape[-1]))
      assert len(emb_dims) == 1, 'all embedding dim must be the same in FM layer:' + ','.join([str(d) for d in emb_dims])
      num_fea = len(inputs)
      emb_dim = emb_dims.pop()
      fea = tf.concat(inputs, axis=-1)
      fea = tf.reshape(fea, [-1, num_fea, emb_dim])
    else:
      assert inputs.shape.ndims == 3, 'input of FM layer must be a 3D tensor or a list of 2D tensors'
      fea = inputs

    square_of_sum = tf.square(tf.reduce_sum(fea, axis=1, keepdims=True))
    sum_of_square = tf.reduce_sum(fea * fea, axis=1, keepdims=True)
    cross_term = square_of_sum - sum_of_square
    cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
    return cross_term
