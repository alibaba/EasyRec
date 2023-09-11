# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.activation import get_activation


class BSTCTR(Layer):

    def __init__(self, params, name='bst_cl4ctr', l2_reg=None, **kwargs):
        super(BSTCTR, self).__init__(name=name, **kwargs)
        self.l2_reg = l2_reg
        self.config = params.get_pb_config()

    def encode(self, fea_input, max_position):
        fea_input = multihead_cross_attention.embedding_postprocessor(
            fea_input,
            position_embedding_name=self.name + '/position_embeddings',
            max_position_embeddings=max_position,
            reuse_position_embedding=tf.AUTO_REUSE)

        n = tf.count_nonzero(fea_input, axis=-1)
        seq_mask = tf.cast(n > 0, tf.int32)

        attention_mask = multihead_cross_attention.create_attention_mask_from_input_mask(
            from_tensor=fea_input, to_mask=seq_mask)

        hidden_act = get_activation(self.config.hidden_act)
        attention_fea = multihead_cross_attention.transformer_encoder(
            fea_input,
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
            reuse=tf.AUTO_REUSE)
        # attention_fea shape: [batch_size, num_features, hidden_size]
        # out_fea shape: [batch_size * num_features, hidden_size]
        out_fea = tf.reshape(attention_fea, [-1, self.config.hidden_size])
        print('bst_cl4ctr output shape:', out_fea.shape)
        return out_fea

    def call(self, inputs, training=None, **kwargs):
        # inputs: [batch_size, num_features, embed_size]
        if not training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0
        max_position = self.config.max_position_embeddings

        return self.encode(inputs, max_position)
