# -*- encoding:utf-8 -*-
from __future__ import print_function

import tensorflow as tf
# from tensorflow.contrib import layers
from tensorflow.python.ops import variable_scope

from easy_rec.python.layers import dnn

# from easy_rec.python.layers.sam_attention import memory
# from easy_rec.python.utils.shape_utils import get_shape_list


class SAM(object):

  def __init__(self, config, l2_reg, is_training, name='sam', **kwargs):
    # super(DIN, self).__init__(name=name, **kwargs)
    self.name = name
    self.l2_reg = l2_reg
    self.config = config
    self._is_training = is_training

  def _target_attention(self, target_feature, keys, seq_mask, mem, name=''):
    target_emb_size = target_feature.get_shape()[-1]
    seq_emb_size = keys.get_shape()[-1]

    if target_emb_size < seq_emb_size:
      pad_size = seq_emb_size - target_emb_size
      query = tf.pad(target_feature, [[0, 0], [0, pad_size]])
      if mem is not None:
        mem = tf.pad(mem, [[0, 0], [0, pad_size]])
    else:
      assert False, 'the embedding size of target item is larger than the one of sequence'

    max_seq_len = tf.shape(keys)[1]
    queries = tf.tile(tf.expand_dims(query, 1), [1, max_seq_len, 1])
    if mem is not None:
      mem = tf.tile(tf.expand_dims(mem, 1), [1, max_seq_len, 1])
    if mem is not None:
      din_all = tf.concat([
          queries[:, :, :target_emb_size], mem[:, :, :target_emb_size], keys,
          queries - keys, queries * keys, mem - keys, mem * keys
      ],
                          axis=-1)  # noqa: E126
    else:
      din_all = tf.concat([
          queries[:, :, :target_emb_size], keys, queries - keys, queries * keys
      ],
                          axis=-1)  # noqa: E126
    din_layer = dnn.DNN(
        self.config.attention_dnn,
        self.l2_reg,
        name + '/target_attention',
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    output = din_layer(din_all)  # [B, L, 1]
    scores = tf.transpose(output, [0, 2, 1])  # [B, 1, L]

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
    # if self.config.need_target_feature:
    #   output = tf.concat([output, target_feature], axis=-1)
    return output, scores

  def _avg_entropy_per_batch(self, att_vec):
    att_vec_sum = tf.add(tf.reduce_sum(att_vec, axis=1), tf.constant(1e-12))
    att_vec_norm = att_vec / tf.reshape(att_vec_sum, (-1, 1))
    # bool_mask = tf.not_equal(att_vec_norm, 0)
    att_vec_entropy = -tf.reduce_sum(
        att_vec_norm * (tf.log(att_vec_norm + 1e-12) / tf.log(2.0)), axis=1)
    avg_entropy_per_batch = tf.reduce_mean(att_vec_entropy)
    return avg_entropy_per_batch

  def __call__(self, inputs, training=None, **kwargs):
    seq_features, target_feature = inputs
    seq_input = [seq_fea for seq_fea, _ in seq_features]
    keys = tf.concat(seq_input, axis=-1)

    # initialize to query layer
    target_emb_size = target_feature.get_shape()[-1]
    mem_cell = tf.nn.rnn_cell.GRUCell(num_units=target_emb_size)
    mem_out = None

    max_seq_len = tf.shape(keys)[1]
    seq_len = seq_features[0][1]
    seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.bool)
    seq_mask = tf.expand_dims(seq_mask, 1)

    outputs = [target_feature]
    with variable_scope.variable_scope(self.name + '/sam'):
      # (?, T_q, C), (?, T_q, T_k
      for i in range(self.config.num_mem_pass):
        attn_out, attn_w = self._target_attention(
            target_feature,
            keys,
            seq_mask,
            mem_out,
            name='target_attention_%d' % i)

        mem_out = target_feature if mem_out is None else mem_out
        mem_out, _ = mem_cell(attn_out, mem_out)  # mem out is initial state
        outputs.extend([attn_out, mem_out])
    output = tf.concat(outputs, axis=-1)
    if not self.config.gru_out:
      #   return layers.batch_norm(
      #       attention_output_layer, scale=True, is_training=self._is_training)
      return output
    else:
      with variable_scope.variable_scope('gru_out'):
        output_cell = tf.nn.rnn_cell.GRUCell(num_units=target_emb_size)
        state = mem_out
        for i in range(3):
          trans_state = tf.layers.dense(
              inputs=state,
              units=target_emb_size,
              activation=None,
              name='dnn',
              reuse=tf.AUTO_REUSE)
          concat_layer = tf.concat([trans_state, target_feature], axis=-1)
          output, state = output_cell(concat_layer, state)
      return output
