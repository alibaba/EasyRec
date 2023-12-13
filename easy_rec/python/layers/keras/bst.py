# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.activation import get_activation
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class BST(Layer):

  def __init__(self, params, name='bst', reuse=None, **kwargs):
    super(BST, self).__init__(name=name, **kwargs)
    self.reuse = reuse
    self.l2_reg = params.l2_regularizer
    self.config = params.get_pb_config()

  def encode(self, seq_input, max_position):
    seq_fea = multihead_cross_attention.embedding_postprocessor(
        seq_input,
        position_embedding_name=self.name,
        max_position_embeddings=max_position,
        reuse_position_embedding=self.reuse)

    n = tf.count_nonzero(seq_input, axis=-1)
    seq_mask = tf.cast(n > 0, tf.int32)

    attention_mask = multihead_cross_attention.create_attention_mask_from_input_mask(
        from_tensor=seq_fea, to_mask=seq_mask)

    hidden_act = get_activation(self.config.hidden_act)
    attention_fea = multihead_cross_attention.transformer_encoder(
        seq_fea,
        hidden_size=self.config.hidden_size,
        num_hidden_layers=self.config.num_hidden_layers,
        num_attention_heads=self.config.num_attention_heads,
        attention_mask=attention_mask,
        intermediate_size=self.config.intermediate_size,
        intermediate_act_fn=hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        initializer_range=self.config.initializer_range,
        name=self.name + '/transformer',
        reuse=self.reuse)
    # attention_fea shape: [batch_size, seq_length, hidden_size]
    if self.config.output_all_token_embeddings:
      out_fea = tf.reshape(attention_fea,
                           [-1, max_position * self.config.hidden_size])
    else:
      out_fea = attention_fea[:, 0, :]  # target feature
    print('bst output shape:', out_fea.shape)
    return out_fea

  def call(self, inputs, training=None, **kwargs):
    if not training:
      self.config.hidden_dropout_prob = 0.0
      self.config.attention_probs_dropout_prob = 0.0
    assert isinstance(inputs, (list, tuple))
    assert len(inputs) >= 2
    # seq_input: [batch_size, seq_len, embed_size]
    seq_input, seq_len = inputs[:2]
    target = inputs[2] if len(inputs) > 2 else None
    max_position = self.config.max_position_embeddings
    # max_seq_len: the max sequence length in current mini-batch, all sequences are padded to this length
    batch_size, cur_batch_max_seq_len, seq_embed_size = get_shape_list(
        seq_input, 3)
    valid_len = tf.assert_less_equal(
        cur_batch_max_seq_len,
        max_position,
        message='sequence length is greater than `max_position_embeddings`:' +
        str(max_position) + ' in feature group:' + self.name +
        ', you should set `max_seq_len` in sequence feature configs')

    if self.config.output_all_token_embeddings:
      seq_input = tf.cond(
          tf.constant(max_position) > cur_batch_max_seq_len, lambda: tf.pad(
              seq_input, [[0, 0], [0, max_position - cur_batch_max_seq_len],
                          [0, 0]], 'CONSTANT'),
          lambda: tf.slice(seq_input, [0, 0, 0], [-1, max_position, -1]))

    if seq_embed_size != self.config.hidden_size:
      seq_input = tf.layers.dense(
          seq_input,
          self.config.hidden_size,
          activation=tf.nn.relu,
          kernel_regularizer=self.l2_reg,
          name=self.name + '/seq_project',
          reuse=self.reuse)

    keep_target = self.config.target_item_position in ('head', 'tail')
    if target is not None and keep_target:
      target_size = target.shape.as_list()[-1]
      assert seq_embed_size == target_size, 'the embedding size of sequence and target item is not equal' \
                                            ' in feature group:' + self.name
      if target_size != self.config.hidden_size:
        target = tf.layers.dense(
            target,
            self.config.hidden_size,
            activation=tf.nn.relu,
            kernel_regularizer=self.l2_reg,
            name=self.name + '/target_project',
            reuse=self.reuse)
      # target_feature: [batch_size, 1, embed_size]
      target = tf.expand_dims(target, 1)
      # seq_input: [batch_size, seq_len+1, embed_size]
      if self.config.target_item_position == 'head':
        seq_input = tf.concat([target, seq_input], axis=1)
      else:
        seq_input = tf.concat([seq_input, target], axis=1)
      max_position += 1
    elif self.config.reserve_target_position:
      max_position += 1

    with tf.control_dependencies([valid_len]):
      return self.encode(seq_input, max_position)
