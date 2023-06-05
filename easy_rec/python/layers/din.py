# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.utils.shape_utils import get_shape_list

# from tensorflow.python.keras.layers import Layer


class DIN(object):

  def __init__(self, config, l2_reg, name='din', **kwargs):
    # super(DIN, self).__init__(name=name, **kwargs)
    self.name = name
    self.l2_reg = l2_reg
    self.config = config

  def __call__(self, inputs, training=None, **kwargs):
    seq_features, target_feature = inputs
    seq_input = [seq_fea for seq_fea, _ in seq_features]
    keys = tf.concat(seq_input, axis=-1)

    query = target_feature
    target_emb_size = target_feature.shape.as_list()[-1]
    seq_emb_size = keys.shape.as_list()[-1]
    if target_emb_size != seq_emb_size:
      logging.info(
          '<din> the embedding size of sequence [%d] and target item [%d] is not equal'
          ' in feature group: %s', seq_emb_size, target_emb_size, self.name)
      if target_emb_size < seq_emb_size:
        query = tf.pad(target_feature,
                       [[0, 0], [0, seq_emb_size - target_emb_size]])
      else:
        assert False, 'the embedding size of target item is larger than the one of sequence'

    batch_size, max_seq_len, _ = get_shape_list(keys, 3)
    queries = tf.tile(tf.expand_dims(query, 1), [1, max_seq_len, 1])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys],
                        axis=-1)
    din_layer = dnn.DNN(
        self.config.attention_dnn,
        self.l2_reg,
        self.name + '/din_attention',
        training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    output = din_layer(din_all)  # [B, L, 1]
    scores = tf.transpose(output, [0, 2, 1])  # [B, 1, L]

    seq_len = seq_features[0][1]
    seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.bool)
    seq_mask = tf.expand_dims(seq_mask, 1)
    paddings = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(seq_mask, scores, paddings)  # [B, 1, L]
    if self.config.attention_normalizer == 'softmax':
      scores = tf.nn.softmax(scores)  # (B, 1, L)
    elif self.config.attention_normalizer == 'softmax_eps':
      scores = scores - tf.reduce_max(scores, axis=2, keepdims=True)
      scores = tf.math.exp(scores)
      scores = scores / (
          tf.reduce_sum(scores, axis=2, keepdims=True) +
          self.config.softmax_eps)
    elif self.config.attention_normalizer == 'sigmoid':
      scores = scores / (seq_emb_size**0.5)
      scores = tf.nn.sigmoid(scores)
    else:
      raise ValueError('unsupported attention normalizer: ' +
                       self.config.attention_normalizer)

    if target_emb_size < seq_emb_size:
      keys = keys[:, :, :target_emb_size]  # [B, L, E]
    output = tf.squeeze(tf.matmul(scores, keys), axis=[1])
    if self.config.need_target_feature:
      output = tf.concat([output, target_feature], axis=-1)
    print('din output shape:', output.shape)
    return output
