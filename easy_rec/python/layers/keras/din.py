# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras import MLP
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.utils.shape_utils import get_shape_list


class DIN(Layer):

  def __init__(self, params, name='din', reuse=None, **kwargs):
    super(DIN, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self.l2_reg = params.l2_regularizer
    self.config = params.get_pb_config()
    self.config.attention_dnn.use_final_bn = False
    self.config.attention_dnn.use_final_bias = True
    self.config.attention_dnn.final_activation = 'linear'
    mlp_params = Parameter.make_from_pb(self.config.attention_dnn)
    mlp_params.l2_regularizer = self.l2_reg
    self.din_layer = MLP(mlp_params, 'din_attention', reuse=self.reuse)

  def call(self, inputs, training=None, **kwargs):
    keys, seq_len, query = inputs
    assert query is not None, '[%s] target feature is empty' % self.name
    query_emb_size = int(query.shape[-1])
    seq_emb_size = keys.shape.as_list()[-1]
    if query_emb_size != seq_emb_size:
      logging.info(
          '<din> the embedding size of sequence [%d] and target item [%d] is not equal'
          ' in feature group: %s', seq_emb_size, query_emb_size, self.name)
      if query_emb_size < seq_emb_size:
        query = tf.pad(query, [[0, 0], [0, seq_emb_size - query_emb_size]])
      else:
        assert False, 'the embedding size of target item is larger than the one of sequence'

    batch_size, max_seq_len, _ = get_shape_list(keys, 3)
    queries = tf.tile(tf.expand_dims(query, 1), [1, max_seq_len, 1])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys],
                        axis=-1)
    output = self.din_layer(din_all, training)  # [B, L, 1]
    scores = tf.transpose(output, [0, 2, 1])  # [B, 1, L]

    seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.bool)
    seq_mask = tf.expand_dims(seq_mask, 1)
    paddings = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(seq_mask, scores, paddings)  # [B, 1, L]
    if self.config.attention_normalizer == 'softmax':
      scores = tf.nn.softmax(scores)  # (B, 1, L)
    elif self.config.attention_normalizer == 'sigmoid':
      scores = scores / (seq_emb_size**0.5)
      scores = tf.nn.sigmoid(scores)
    else:
      raise ValueError('unsupported attention normalizer: ' +
                       self.config.attention_normalizer)

    if query_emb_size < seq_emb_size:
      keys = keys[:, :, :query_emb_size]  # [B, L, E]
    output = tf.squeeze(tf.matmul(scores, keys), axis=[1])
    if self.config.need_target_feature:
      output = tf.concat([output, query], axis=-1)
    print('din output shape:', output.shape)
    return output
