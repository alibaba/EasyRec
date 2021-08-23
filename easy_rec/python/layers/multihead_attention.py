# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MultiHeadAttention:

  def __init__(self, head_num, head_size, l2_reg, use_res=False, name=''):
    """Initializes a `MultiHeadAttention` Layer.

    Args:
      head_num: The number of heads
      head_size: The dimension of a head
      l2_reg: l2 regularizer
      use_res: Whether to use residual connections before output.
      name: scope of the MultiHeadAttention, so that the parameters could be separated from other MultiHeadAttention
    """
    self._head_num = head_num
    self._head_size = head_size
    self._l2_reg = l2_reg
    self._use_res = use_res
    self._name = name

  def _split_multihead_qkv(self, q, k, v):
    """Split multiple heads.

    Args:
      q: Query matrix of shape [bs, feature_num, head_num * head_size].
      k: Key matrix of shape [bs, feature_num, head_num * head_size].
      v: Value matrix of shape [bs, feature_num, head_num * head_size].

    Returns:
      q: Query matrix of shape [bs, head_num, feature_num, head_size].
      k: Key matrix of shape [bs, head_num, feature_num, head_size].
      v: Value matrix of shape [bs, head_num, feature_num, head_size].
    """
    reshaped_q = tf.reshape(
        q, shape=[-1, q.shape[1], self._head_num, self._head_size])
    q = tf.transpose(reshaped_q, perm=[0, 2, 1, 3])
    reshaped_k = tf.reshape(
        k, shape=[-1, k.shape[1], self._head_num, self._head_size])
    k = tf.transpose(reshaped_k, perm=[0, 2, 1, 3])
    reshaped_v = tf.reshape(
        v, shape=[-1, v.shape[1], self._head_num, self._head_size])
    v = tf.transpose(reshaped_v, perm=[0, 2, 1, 3])
    return q, k, v

  def _scaled_dot_product_attention(self, q, k, v):
    """Calculate scaled dot product attention by q, k and v.

    Args:
      q: Query matrix of shape [bs, head_num, feature_num, head_size].
      k: Key matrix of shape [bs, head_num, feature_num, head_size].
      v: Value matrix of shape [bs, head_num, feature_num, head_size].

    Returns:
      q: Query matrix of shape [bs, head_num, feature_num, head_size].
      k: Key matrix of shape [bs, head_num, feature_num, head_size].
      v: Value matrix of shape [bs, head_num, feature_num, head_size].
    """
    product = tf.linalg.matmul(
        a=q, b=k, transpose_b=True) / (
            self._head_size**-0.5)
    weights = tf.nn.softmax(product)
    out = tf.linalg.matmul(weights, v)
    return out

  def _compute_qkv(self, q, k, v):
    """Calculate q, k and v matrices.

    Args:
      q: Query matrix of shape [bs, feature_num, d_model].
      k: Key matrix of shape [bs, feature_num, d_model].
      v: Value matrix of shape [bs, feature_num, d_model].

    Returns:
      q: Query matrix of shape [bs, feature_num, head_size * n_head].
      k: Key matrix of shape [bs, feature_num, head_size * n_head].
      v: Value matrix of shape [bs, feature_num, head_size * n_head].
    """
    q = tf.layers.dense(
        q,
        self._head_num * self._head_size,
        use_bias=False,
        kernel_regularizer=self._l2_reg,
        name='%s/%s/dnn' % (self._name, 'query'))
    k = tf.layers.dense(
        k,
        self._head_num * self._head_size,
        use_bias=False,
        kernel_regularizer=self._l2_reg,
        name='%s/%s/dnn' % (self._name, 'key'))
    v = tf.layers.dense(
        v,
        self._head_num * self._head_size,
        use_bias=False,
        kernel_regularizer=self._l2_reg,
        name='%s/%s/dnn' % (self._name, 'value'))
    return q, k, v

  def _combine_heads(self, multi_head_tensor):
    """Combine the results of multiple heads.

    Args:
      multi_head_tensor: Result matrix of shape [bs, head_num, feature_num, head_size].

    Returns:
      out: Result matrix of shape [bs, feature_num, head_num * head_size].
    """
    x = tf.transpose(multi_head_tensor, perm=[0, 2, 1, 3])
    out = tf.reshape(x, shape=[-1, x.shape[1], x.shape[2] * x.shape[3]])
    return out

  def _multi_head_attention(self, attention_input):
    """Build multiple heads attention layer.

    Args:
      attention_input: The input of interacting layer, has a shape of [bs, feature_num, d_model].

    Returns:
      out: The output of multi head attention layer, has a shape of [bs, feature_num, head_num * head_size].
    """
    if isinstance(attention_input, list):
      assert len(attention_input) == 3 or len(attention_input) == 1, \
          'If the input of multi_head_attention is a list, the length must be 1 or 3.'

      if len(attention_input) == 3:
        ori_q = attention_input[0]
        ori_k = attention_input[1]
        ori_v = attention_input[2]
      else:
        ori_q = attention_input[0]
        ori_k = attention_input[0]
        ori_v = attention_input[0]
    else:
      ori_q = attention_input
      ori_k = attention_input
      ori_v = attention_input

    q, k, v = self._compute_qkv(ori_q, ori_k, ori_v)
    q, k, v = self._split_multihead_qkv(q, k, v)
    multi_head_tensor = self._scaled_dot_product_attention(q, k, v)
    out = self._combine_heads(multi_head_tensor)

    if self._use_res:
      W_0_x = tf.layers.dense(
          ori_v,
          out.shape[2],
          use_bias=False,
          kernel_regularizer=self._l2_reg,
          name='%s/dnn' % (self._name))
      res_out = tf.nn.relu(out + W_0_x)
      return res_out
    else:
      return out

  def __call__(self, deep_fea):
    deep_fea = self._multi_head_attention(deep_fea)
    return deep_fea
