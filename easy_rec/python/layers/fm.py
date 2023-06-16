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
      fm_feas = tf.stack(fm_fea, axis=1)
      sum_square = tf.square(tf.reduce_sum(fm_feas, 1))
      square_sum = tf.reduce_sum(tf.square(fm_feas), 1)
      y_v = 0.5 * tf.subtract(sum_square, square_sum)
    return y_v


class FMLayer(object):
  """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

  References
    - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  """

  def __init__(self, config, name='fm'):
    self.name = name
    self.config = config

  def __call__(self, inputs):
    """FM layer.

    Input shape.
      - List of 2D tensor with shape: ``(batch_size,embedding_size)``.
      - Or a 3D tensor with shape: ``(batch_size,field_size,embedding_size)``
    Output shape
      - 2D tensor with shape: ``(batch_size, 1)``.
    """
    if type(inputs) == list:
      emb_dims = set(map(lambda x: int(x.shape[-1]), inputs))
      if len(emb_dims) != 1:
        dims = ','.join([str(d) for d in emb_dims])
        raise ValueError('all embedding dim must be equal in FM layer:' + dims)

      with tf.name_scope(self.name):
        fea = tf.stack(inputs, axis=1)
    else:
      assert inputs.shape.ndims == 3, 'input of FM layer must be a 3D tensor or a list of 2D tensors'
      fea = inputs

    with tf.name_scope(self.name):
      square_of_sum = tf.square(tf.reduce_sum(fea, axis=1))
      sum_of_square = tf.reduce_sum(tf.square(fea), axis=1)
      cross_term = tf.subtract(square_of_sum, sum_of_square)
      if self.config.use_variant:
        cross_term = 0.5 * cross_term
      else:
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=-1)
    return cross_term
