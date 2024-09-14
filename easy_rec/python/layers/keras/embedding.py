# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fused embedding layer."""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class EmbeddingLayer(Layer):

  def __init__(self, params, name='embedding_layer', reuse=None, **kwargs):
    super(EmbeddingLayer, self).__init__(name=name, **kwargs)
    params.check_required(['vocab_size', 'embedding_dim'])
    vocab_size = int(params.vocab_size)
    embed_dim = int(params.embedding_dim)
    self.embedding_table = self.add_weight(
        shape=(vocab_size, embed_dim),
        initializer='random_normal',
        trainable=True,
        name='embedding_table')
    self.do_concat = params.get_or_default('concat', True)

  def call(self, inputs, training=None, **kwargs):
    # 将多个特征的输入合并为一个索引 tensor
    all_indices = tf.concat(
        [tf.reshape(input_field, [-1]) for input_field in inputs], axis=0)

    # 从共享的嵌入表中进行一次 embedding lookup
    all_embeddings = tf.nn.embedding_lookup(self.embedding_table, all_indices)

    # 计算每个特征的嵌入
    split_sizes = []
    for input_field in inputs:
      size = tf.shape(input_field)[0]
      if input_field.ndim > 1:
        size *= input_field.shape[-1]
      split_sizes.append(size)
    embeddings = tf.split(all_embeddings, split_sizes, axis=0)
    if self.do_concat:
      return tf.concat(embeddings, axis=-1)
    return embeddings
