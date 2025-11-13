# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.loss import contrastive_loss


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

    return loss_value
