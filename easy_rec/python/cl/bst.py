# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.activation import get_activation
from easy_rec.python.utils.shape_utils import get_shape_list
from easy_rec.python.cl.augment import input_aug_data
from easy_rec.python.cl.loss import nce_loss


# def input_data(self, seq_features, target_feature):
#     seq_embeds = [seq_fea for seq_fea, _ in seq_features]
#     max_position = self.config.max_position_embeddings
#     # max_seq_len: the max sequence length in current mini-batch, all sequences are padded to this length
#     batch_size, max_seq_len, _ = get_shape_list(seq_features[0][0], 3)
#     valid_len = tf.assert_less_equal(
#         max_seq_len,
#         max_position,
#         message='sequence length is greater than `max_position_embeddings`:' +
#                 str(max_position) + ' in feature group:' + self.name)
#     with tf.control_dependencies([valid_len]):
#         # seq_input: [batch_size, seq_len, embed_size]
#         seq_input = tf.concat(seq_embeds, axis=-1)
#
#     # seq_len: [batch_size, 1], the true length of each sequence
#     seq_len = seq_features[0][1]
#     seq_embed_size = seq_input.shape.as_list()[-1]
#     if target_feature is not None:
#         target_size = target_feature.shape.as_list()[-1]
#         assert seq_embed_size == target_size, 'the embedding size of sequence and target item is not equal' \
#                                               ' in feature group:' + self.name
#         # target_feature: [batch_size, 1, embed_size]
#         target_feature = tf.expand_dims(target_feature, 1)
#         # seq_input: [batch_size, seq_len+1, embed_size]
#         seq_input = tf.concat([target_feature, seq_input], axis=1)
#         max_seq_len += 1
#         seq_len += 1
#         max_position += 1
#
#     if seq_embed_size != self.config.hidden_size:
#         seq_input = tf.layers.dense(
#             seq_input,
#             self.config.hidden_size,
#             activation=tf.nn.relu,
#             kernel_regularizer=self.l2_reg)
#     return seq_input, seq_len, max_position, max_seq_len


def model(self, seq_input, seq_len, max_position, max_seq_len):
    seq_fea = multihead_cross_attention.embedding_postprocessor(
        seq_input,
        position_embedding_name=self.name + '/position_embeddings',
        max_position_embeddings=max_position,
        reuse_position_embedding=tf.AUTO_REUSE)
    seq_mask = tf.map_fn(
        fn=lambda t: dynamic_mask(t, max_seq_len), elems=tf.to_int32(seq_len))
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
        name=self.name + '/bst',
        reuse=tf.AUTO_REUSE)
    # attention_fea shape: [batch_size, seq_length, hidden_size]
    out_fea = attention_fea[:, 0, :]  # target feature
    print('bst output shape:', out_fea.shape)
    return out_fea


def dynamic_mask(x, max_len):
    ones = tf.ones(shape=tf.stack([x]), dtype=tf.int32)
    zeros = tf.zeros(shape=tf.stack([max_len - x]), dtype=tf.int32)
    return tf.concat([ones, zeros], axis=0)


class BST(object):

    def __init__(self, config, l2_reg, name='bst', **kwargs):
        # super(BST, self).__init__(name=name, **kwargs)
        self.name = name
        self.l2_reg = l2_reg
        self.config = config

    def cl_bst(self, seq_input, seq_len, max_position, max_seq_len):
        aug_seq1, aug_seq2, aug_len1, aug_len2 = input_aug_data(seq_input, seq_len)
        seq_output1 = model(self, aug_seq1, aug_len1, max_position, max_seq_len)
        seq_output2 = model(self, aug_seq2, aug_len2, max_position, max_seq_len)
        loss = nce_loss(seq_output1, seq_output2)
        return loss

    def __call__(self, inputs, training=None, **kwargs):
        seq_features, target_feature = inputs
        if not training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0

        seq_embeds = [seq_fea for seq_fea, _ in seq_features]
        max_position = self.config.max_position_embeddings
        # max_seq_len: the max sequence length in current mini-batch, all sequences are padded to this length
        batch_size, max_seq_len, _ = get_shape_list(seq_features[0][0], 3)
        valid_len = tf.assert_less_equal(
            max_seq_len,
            max_position,
            message='sequence length is greater than `max_position_embeddings`:' +
                    str(max_position) + ' in feature group:' + self.name)
        with tf.control_dependencies([valid_len]):
            # seq_input: [batch_size, seq_len, embed_size]
            seq_input = tf.concat(seq_embeds, axis=-1)

        # seq_len: [batch_size, 1], the true length of each sequence
        seq_len = seq_features[0][1]
        seq_embed_size = seq_input.shape.as_list()[-1]

        if seq_embed_size != self.config.hidden_size:
            seq_input = tf.layers.dense(
                seq_input,
                self.config.hidden_size,
                activation=tf.nn.relu,
                kernel_regularizer=self.l2_reg)

        if target_feature is not None:
            max_position += 1

        if self.config.need_cl:
            assert 'loss_dict' in kwargs, "no `loss_dict` in kwargs"
            loss = self.cl_bst(seq_input, seq_len, max_position, max_seq_len)
            loss_dict = kwargs['loss_dict']
            loss_dict['cl_loss'] = loss

        if target_feature is not None:
            target_size = target_feature.shape.as_list()[-1]
            assert seq_embed_size == target_size, 'the embedding size of sequence and target item is not equal' \
                                                  ' in feature group:' + self.name
            if target_size != self.config.hidden_size:
                target_feature = tf.layers.dense(
                    target_feature,
                    self.config.hidden_size,
                    activation=tf.nn.relu,
                    kernel_regularizer=self.l2_reg)

            # target_feature: [batch_size, 1, embed_size]
            target_feature = tf.expand_dims(target_feature, 1)
            # seq_input: [batch_size, seq_len+1, embed_size]
            seq_input = tf.concat([target_feature, seq_input], axis=1)
            max_seq_len += 1
            seq_len += 1

        # seq_input, max_position, max_seq_len, seq_len = input_data(self, seq_features, target_feature)
        seq_output = model(self, seq_input, seq_len, max_position, max_seq_len)
        return seq_output
