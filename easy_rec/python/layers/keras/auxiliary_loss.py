# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.loss import contrastive_loss
from easy_rec.python.utils.shape_utils import get_shape_list


class AuxiliaryLoss(tf.keras.layers.Layer):
    """Compute auxiliary loss, usually use for contrastive learning."""

    def __init__(self, params, name='auxiliary_loss', reuse=None, **kwargs):
        super(AuxiliaryLoss, self).__init__(name=name, **kwargs)
        params.check_required('loss_type')
        self.loss_type = params.get_or_default('loss_type', None)
        self.loss_weight = params.get_or_default('loss_weight', 1.0)
        logging.info('init layer `%s` with loss type: %s and weight: %f' %
                     (self.name, self.loss_type, self.loss_weight))
        self.temperature = params.get_or_default('temperature', 0.1)

    def call(self, inputs, training=None, **kwargs):
        if self.loss_type is None:
            logging.warning('loss_type is None in auxiliary loss layer')
            return 0

        loss_dict = kwargs['loss_dict']
        loss_value = 0

        if self.loss_type == 'l2_loss':
            x1, x2 = inputs
            loss = contrastive_loss.l2_loss(x1, x2)
            loss_value = loss if self.loss_weight == 1.0 else loss * self.loss_weight
            loss_dict['%s_l2_loss' % self.name] = loss_value
        elif self.loss_type == 'info_nce':
            query, positive = inputs
            loss = contrastive_loss.info_nce_loss(
                query, positive, temperature=self.temperature)
            loss_value = loss if self.loss_weight == 1.0 else loss * self.loss_weight
            loss_dict['%s_info_nce_loss' % self.name] = loss_value
        elif self.loss_type == 'nce_loss':
            x1, x2 = inputs
            loss = contrastive_loss.nce_loss(x1, x2, temperature=self.temperature)
            loss_value = loss if self.loss_weight == 1.0 else loss * self.loss_weight
            loss_dict['%s_nce_loss' % self.name] = loss_value

        elif self.loss_type == 'alignment_loss':
            fea_emb = inputs
            batch_size = get_shape_list(fea_emb)[0]
            # indices = tf.where(tf.ones([tf.reduce_sum(batch_size), tf.reduce_sum(batch_size)]))
            indices = tf.where(tf.ones(tf.stack([batch_size, batch_size])))
            row = tf.gather(tf.reshape(indices[:, 0], [-1]), tf.where(indices[:, 0] < indices[:, 1]))
            col = tf.gather(tf.reshape(indices[:, 1], [-1]), tf.where(indices[:, 0] < indices[:, 1]))
            row = tf.squeeze(row)
            col = tf.squeeze(col)
            x_row = tf.gather(fea_emb, row)
            x_col = tf.gather(fea_emb, col)
            distance_sq = tf.reduce_sum(tf.square(tf.subtract(x_row, x_col)), axis=2)
            alignment_loss = tf.reduce_mean(distance_sq)
            loss_value = alignment_loss if self.loss_weight == 1.0 else alignment_loss * self.loss_weight
            loss_dict['%s_alignment_loss' % self.name] = loss_value

        return loss_value
