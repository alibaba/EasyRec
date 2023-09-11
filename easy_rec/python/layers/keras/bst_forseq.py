# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.activation import get_activation
from easy_rec.python.utils.shape_utils import get_shape_list
from easy_rec.python.loss.nce_loss import nce_loss
from easy_rec.python.input.augment import input_aug_data


# from tensorflow.python.keras.layers import Layer


class BSTSEQ(object):

    def __init__(self, params, name='bstseq', l2_reg=None, **kwargs):
        # super(BSTSEQ, self).__init__(name=name, **kwargs)
        self.name = name
        self.l2_reg = l2_reg
        self.config = params.get_pb_config()

    def encode(self, seq_input, max_position):
        seq_fea = multihead_cross_attention.embedding_postprocessor(
            seq_input,
            position_embedding_name=self.name + '/position_embeddings',
            max_position_embeddings=max_position,
            reuse_position_embedding=tf.AUTO_REUSE)

        n = tf.count_nonzero(seq_fea, axis=-1)
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
            name=self.name + '/bstseq',
            reuse=tf.AUTO_REUSE)
        # attention_fea shape: [batch_size, seq_length, hidden_size]
        out_fea = attention_fea[:, 0, :]  # target feature
        return out_fea

    def __call__(self, inputs, training=None, **kwargs):
        seq_features, _ = inputs
        assert len(seq_features) > 0, '[%s] sequence feature is empty' % self.name
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

        seq_embed_size = seq_input.shape.as_list()[-1]
        if seq_embed_size != self.config.hidden_size:
            seq_input = tf.layers.dense(
                seq_input,
                self.config.hidden_size,
                activation=tf.nn.relu,
                kernel_regularizer=self.l2_reg)

        seq_len = seq_features[0][1]
        if self.config.need_contrastive_learning:
            with tf.variable_scope('cl_mask', reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('mask_tensor',
                                          shape=[seq_input.shape.as_list()[-1]],
                                          trainable=True,
                                          initializer=tf.truncated_normal_initializer(stddev=0.02))
            cl_loss = self.contrastive_loss(seq_input, seq_len, max_position, weights, self.config.cl_param)
            cl_loss *= self.config.contrastive_loss_weight
            assert 'loss_dict' in kwargs, "no `loss_dict` in kwargs of bst layer: %s" % self.name
            loss_dict = kwargs['loss_dict']
            loss_dict['%s_contrastive_loss' % self.name] = cl_loss
        return self.encode(seq_input, max_position)

    def contrastive_loss(self, seq_input, seq_len, max_position, weights, cl_param):
        aug_seq1, aug_seq2, aug_len1, aug_len2 = input_aug_data(seq_input, seq_len, weights, cl_param)
        seq_output1 = self.encode(aug_seq1, max_position)
        seq_output2 = self.encode(aug_seq2, max_position)
        loss = nce_loss(seq_output1, seq_output2)
        return loss
