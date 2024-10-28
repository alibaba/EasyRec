# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fused embedding layer."""
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Layer


def _combine(embeddings, weights, comb_fn):
  # embeddings shape: [B, N, D]
  if callable(comb_fn):
    return comb_fn(embeddings, axis=1)
  if weights is None:
    return tf.reduce_mean(embeddings, axis=1)
  if isinstance(weights, tf.SparseTensor):
    if weights.dtype == tf.string:
      weights = tf.sparse.to_dense(weights, default_value='0')
      weights = tf.string_to_number(weights)
    else:
      weights = tf.sparse.to_dense(weights, default_value=0.0)
  sum_weights = tf.reduce_sum(weights, axis=1, keepdims=True)
  weights = tf.expand_dims(weights / sum_weights, axis=-1)
  return tf.reduce_sum(embeddings * weights, axis=1)


class EmbeddingLayer(Layer):

  def __init__(self, params, name='embedding_layer', reuse=None, **kwargs):
    super(EmbeddingLayer, self).__init__(name=name, **kwargs)
    params.check_required(['vocab_size', 'embedding_dim'])
    vocab_size = int(params.vocab_size)
    combiner = params.get_or_default('combiner', 'weight')
    if combiner == 'mean':
      self.combine_fn = tf.reduce_mean
    elif combiner == 'sum':
      self.combine_fn = tf.reduce_sum
    elif combiner == 'max':
      self.combine_fn = tf.reduce_max
    elif combiner == 'min':
      self.combine_fn = tf.reduce_min
    elif combiner == 'weight':
      self.combine_fn = 'weight'
    else:
      raise ValueError('unsupported embedding combiner: ' + combiner)
    self.embed_dim = int(params.embedding_dim)
    self.embedding = Embedding(vocab_size, self.embed_dim)
    self.do_concat = params.get_or_default('concat', True)

  def call(self, inputs, training=None, **kwargs):
    inputs, weights = inputs
    # 将多个特征的输入合并为一个索引 tensor
    flat_inputs = [tf.reshape(input_field, [-1]) for input_field in inputs]
    all_indices = tf.concat(flat_inputs, axis=0)
    # 从共享的嵌入表中进行一次 embedding lookup
    all_embeddings = self.embedding(all_indices)
    is_multi = []
    # 计算每个特征的嵌入
    split_sizes = []
    for input_field in inputs:
      assert input_field.shape.ndims <= 2, 'dims of embedding layer input must be <= 2'
      input_shape = tf.shape(input_field)
      size = input_shape[0]
      if input_field.shape.ndims > 1:
        size *= input_shape[-1]
        is_multi.append(True)
      else:
        is_multi.append(False)
      split_sizes.append(size)
    embeddings = tf.split(all_embeddings, split_sizes, axis=0)
    for i in range(len(embeddings)):
      if is_multi[i]:
        batch_size = tf.shape(inputs[i])[0]
        embeddings[i] = tf.cond(
            tf.equal(tf.size(embeddings[i]), 0),
            lambda: tf.zeros([batch_size, self.embed_dim]), lambda: _combine(
                tf.reshape(embeddings[i], [batch_size, -1, self.embed_dim]),
                weights[i], self.combine_fn))
    if self.do_concat:
      embeddings = tf.concat(embeddings, axis=-1)
    print('Embedding layer:', self.name, embeddings)
    return embeddings
