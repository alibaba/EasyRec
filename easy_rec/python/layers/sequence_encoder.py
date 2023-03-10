# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.activation import get_activation
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SequenceEncoder(object):

  def __init__(self, input_layer, feature_groups_config, emb_reg, l2_reg):
    self._input_layer = input_layer
    self._feature_groups_config = {
        x.group_name: x for x in feature_groups_config
    }
    self._emb_reg = emb_reg
    self._l2_reg = l2_reg

  def __call__(self, features, group_name, is_training=True, *args, **kwargs):
    group_config = self._feature_groups_config[group_name]
    if len(group_config.sequence_encoders) == 0:
      return None

    seq_features, target_feature, target_features = self._input_layer(
        features, group_name, is_combine=False)
    assert len(
        seq_features) > 0, 'sequence feature is empty in group: ' + group_name

    outputs = []
    for encoder in group_config.sequence_encoders:
      encoder_type = encoder.WhichOneof('encoder').lower()
      if encoder_type == 'bst':
        encoding = self.bst_encoder(seq_features, target_feature, group_name,
                                    encoder.bst, is_training)
        outputs.append(encoding)
      elif encoder_type == 'din':
        encoding = self.din_encoder(seq_features, target_feature, group_name,
                                    encoder.din, is_training)
        outputs.append(encoding)
      else:
        assert False, 'unsupported sequence encode type: ' + encoder_type

    if len(outputs) == 0:
      logging.warning(
          "there's no sequence encoder configured in feature group: " +
          group_name)
      return None
    if len(outputs) == 1:
      return outputs[0]

    return tf.concat(outputs, axis=-1)

  def din_encoder(self, seq_features, target_feature, group_name, config,
                  is_training):
    seq_input = [seq_fea for seq_fea, _ in seq_features]
    regularizers.apply_regularization(self._emb_reg, weights_list=seq_input)
    keys = tf.concat(seq_input, axis=-1)

    target_emb_size = target_feature.shape.as_list()[-1]
    seq_emb_size = keys.shape.as_list()[-1]
    assert target_emb_size == seq_emb_size, 'the embedding size of sequence and target item is not equal' \
                                            ' in feature group:' + group_name

    batch_size, max_seq_len, _ = get_shape_list(keys, 3)
    queries = tf.tile(tf.expand_dims(target_feature, 1), [1, max_seq_len, 1])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys],
                        axis=-1)
    din_layer = dnn.DNN(
        config.attention_dnn,
        self._l2_reg,
        group_name + '/din_attention',
        is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    output = din_layer(din_all)  # [B, L, 1]
    scores = tf.transpose(output, [0, 2, 1])  # [B, 1, L]

    seq_len = seq_features[0][1]
    seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.bool)
    seq_mask = tf.expand_dims(seq_mask, 1)
    paddings = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(seq_mask, scores, paddings)  # [B, 1, L]
    scores = scores / (seq_emb_size**0.5)
    # normalization with softmax is abandoned according to the original paper
    scores = tf.nn.sigmoid(scores)
    output = tf.squeeze(tf.matmul(scores, keys), axis=[1])
    print('din output shape:', output.shape)
    return output

  def bst_encoder(self, seq_features, target_feature, group_name, config,
                  is_training):
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    seq_embeds = [seq_fea for seq_fea, _ in seq_features]
    regularizers.apply_regularization(self._emb_reg, weights_list=seq_embeds)

    max_position = config.max_position_embeddings
    batch_size, max_seq_len, _ = get_shape_list(seq_features[0][0], 3)
    valid_len = tf.assert_less_equal(
        max_seq_len,
        max_position,
        message='sequence length is greater than `max_position_embeddings`:' +
        str(max_position) + ' in feature group:' + group_name)
    with tf.control_dependencies([valid_len]):
      # seq_input: [batch_size, seq_len, embed_size]
      seq_input = tf.concat(seq_embeds, axis=-1)

    # seq_len: [batch_size, ], 假设每个sequence feature的length都是相同的
    seq_len = seq_features[0][1]
    seq_embed_size = seq_input.shape.as_list()[-1]
    if target_feature is not None:
      target_size = target_feature.shape.as_list()[-1]
      assert seq_embed_size == target_size, 'the embedding size of sequence and target item is not equal' \
                                            ' in feature group:' + group_name
      # target_feature: [batch_size, 1, embed_size]
      target_feature = tf.expand_dims(target_feature, 1)
      # seq_input: [batch_size, seq_len+1, embed_size]
      seq_input = tf.concat([target_feature, seq_input], axis=1)
      max_seq_len += 1
      seq_len += 1

    if seq_embed_size != config.hidden_size:
      seq_input = tf.layers.dense(
          seq_input,
          config.hidden_size,
          activation=tf.nn.relu,
          kernel_regularizer=self._l2_reg)

    seq_fea = multihead_cross_attention.embedding_postprocessor(
        seq_input,
        position_embedding_name=group_name + '_position_embeddings',
        max_position_embeddings=max_position)
    seq_mask = tf.map_fn(
        fn=lambda t: dynamic_mask(t, max_seq_len), elems=tf.to_int32(seq_len))
    attention_mask = multihead_cross_attention.create_attention_mask_from_input_mask(
        from_tensor=seq_fea, to_mask=seq_mask)

    hidden_act = get_activation(config.hidden_act)
    attention_fea = multihead_cross_attention.transformer_encoder(
        seq_fea,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        attention_mask=attention_mask,
        intermediate_size=config.intermediate_size,
        intermediate_act_fn=hidden_act,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        name=group_name + '/bst')
    # attention_fea shape: [batch_size, seq_length, hidden_size]
    out_fea = attention_fea[:, 0, :]  # target feature
    print('bst output shape:', out_fea.shape)
    return out_fea


def dynamic_mask(x, max_len):
  ones = tf.ones(shape=tf.stack([x]), dtype=tf.int32)
  zeros = tf.zeros(shape=tf.stack([max_len - x]), dtype=tf.int32)
  return tf.concat([ones, zeros], axis=0)
