# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers.common_layers import layer_norm

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class MaskBlock(object):
  def __init__(self, mask_block_config):
    self.mask_block_config = mask_block_config

  def __call__(self, net, mask_input):
    mask_input_dim = int(mask_input.shape[-1])
    if self.mask_block_config.HasField('reduction_factor'):
      aggregation_size = int(mask_input_dim * self.mask_block_config.reduction_factor)
    elif self.mask_block_config.HasField('aggregation_size') is not None:
      aggregation_size = self.mask_block_config.aggregation_size
    else:
      raise ValueError("Need one of reduction factor or aggregation size for MaskBlock.")

    if self.mask_block_config.input_layer_norm:
      input_name = net.name.replace(':', '_')
      net = layer_norm(net, reuse=tf.AUTO_REUSE, name='ln_' + input_name)

    # initializer = tf.initializers.variance_scaling()
    initializer = tf.glorot_uniform_initializer()
    mask = tf.layers.dense(mask_input, aggregation_size,
                           activation=tf.nn.relu,
                           kernel_initializer=initializer)
    mask = tf.layers.dense(mask, net.shape[-1])
    masked_net = net * mask

    output_size = self.mask_block_config.output_size
    hidden_layer_output = tf.layers.dense(masked_net, output_size)
    return layer_norm(hidden_layer_output)


class MaskNet(object):
  def __init__(self, mask_net_config, name='mask_net'):
    self.mask_net_config = mask_net_config
    self.name = name

  def __call__(self, inputs, is_training, l2_reg=None):
    conf = self.mask_net_config
    if conf.use_parallel:
      mask_outputs = []
      for block_conf in self.mask_net_config.mask_blocks:
        mask_layer = MaskBlock(block_conf)
        mask_outputs.append(mask_layer(mask_input=inputs, net=inputs))
      all_mask_outputs = tf.concat(mask_outputs, axis=1)

      if conf.HasField('mlp'):
        mlp = dnn.DNN(conf.mlp, l2_reg, name='%s/mlp' % self.name, is_training=is_training)
        output = mlp(all_mask_outputs)
      else:
        output = all_mask_outputs
      return output
    else:
      net = inputs
      for block_conf in self.mask_net_config.mask_blocks:
        mask_layer = MaskBlock(block_conf)
        net = mask_layer(net=net, mask_input=inputs)

      if conf.HasField('mlp'):
        mlp = dnn.DNN(conf.mlp, l2_reg, name='%s/mlp' % self.name, is_training=is_training)
        output = mlp(net)
      else:
        output = net
      return output
