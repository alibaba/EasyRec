# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers.common_layers import SENet, EnhancedInputLayer
from easy_rec.python.layers.common_layers import highway, Concatenate
from easy_rec.python.layers.fibinet import FiBiNetLayer
from easy_rec.python.layers.fm import FM, FMLayer
from easy_rec.python.layers.mask_net import MaskNet
from easy_rec.python.layers.numerical_embedding import AutoDisEmbedding
from easy_rec.python.layers.numerical_embedding import PeriodicEmbedding
from easy_rec.python.utils.dag import DAG
from easy_rec.python.utils.tf_utils import add_op, dot_op

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class Backbone(object):

  def __init__(self, config, model, features, input_layer, l2_reg=None):
    self._model = model
    self._config = config
    self._features = features
    self._input_layer = input_layer
    self._l2_reg = l2_reg
    self._dag = DAG()
    self._name_to_blocks = {}
    for block in config.blocks:
      self._name_to_blocks[block.name] = block
      self._dag.add_node(block.name)
    num_blocks = len(self._name_to_blocks)
    assert num_blocks > 0, 'there must be at least one block in backbone'
    for block in config.blocks:
      assert len(block.inputs) > 0, 'no input for block: %s' % block.name
      for node in block.inputs:
        if node in self._name_to_blocks:
          self._dag.add_edge(node, block.name)

  def block_input(self, config, block_outputs, output_list=False):
    inputs = []
    for input_name in config.inputs:
      if input_name in block_outputs:
        input_feature = block_outputs[input_name]
      else:
        input_feature, _ = self._input_layer(self._features, input_name)
      inputs.append(input_feature)

    if output_list:
      output = inputs
    else:
      output = concat_inputs(inputs, config.input_concat_axis, config.name)

    if config.HasField('extra_input_fn'):
      fn = eval(config.extra_input_fn)
      output = fn(output)
    return output

  def __call__(self, is_training, *args, **kwargs):
    block_outputs = {}
    blocks = self._dag.topological_sort()
    logging.info('backbone topological order: ' + ','.join(blocks))
    print('backbone topological order: ' + ','.join(blocks))
    for block in blocks:
      config = self._name_to_blocks[block]
      layer = config.WhichOneof('layer')
      if layer == 'input_layer':
        if len(config.inputs) != 1:
          raise ValueError('only one input allowed for input_layer: ' +
                           block.name)
        conf = config.input_layer
        input_layer = EnhancedInputLayer(conf, self._input_layer,
                                         self._features)
        output = input_layer(config.inputs[0], is_training)
        block_outputs[block] = output
      elif layer == 'periodic_embedding':
        input_feature = self.block_input(config, block_outputs)
        num_emb = PeriodicEmbedding(config.periodic_embedding, scope=block)
        block_outputs[block] = num_emb(input_feature)
      elif layer == 'auto_dis_embedding':
        input_feature = self.block_input(config, block_outputs)
        num_emb = AutoDisEmbedding(config.auto_dis_embedding, scope=block)
        block_outputs[block] = num_emb(input_feature)
      elif layer == 'highway':
        input_feature = self.block_input(config, block_outputs)
        conf = config.highway
        highway_layer = highway(
            input_feature,
            conf.emb_size,
            activation=conf.activation,
            dropout=conf.dropout_rate,
            scope=block)
        block_outputs[block] = highway_layer(input_feature)
      elif layer == 'mlp':
        input_feature = self.block_input(config, block_outputs)
        mlp = dnn.DNN(
            config.mlp,
            self._l2_reg,
            name='%s_mlp' % block,
            is_training=is_training,
            last_layer_no_activation=config.mlp.last_layer_no_activation,
            last_layer_no_batch_norm=config.mlp.last_layer_no_batch_norm)
        block_outputs[block] = mlp(input_feature)
      elif layer == 'sequence_encoder':
        block_outputs[block] = self.sequence_encoder(config, is_training)
      elif layer == 'masknet':
        input_feature = self.block_input(config, block_outputs)
        mask_net = MaskNet(config.masknet, name=block, reuse=tf.AUTO_REUSE)
        output = mask_net(input_feature, is_training, l2_reg=self._l2_reg)
        block_outputs[block] = output
      elif layer == 'senet':
        input_feature = self.block_input(config, block_outputs)
        senet = SENet(config.senet, name=block)
        output = senet(input_feature)
        block_outputs[block] = output
      elif layer == 'fibinet':
        input_feature = self.block_input(config, block_outputs)
        fibinet = FiBiNetLayer(config.fibinet, name=block)
        output = fibinet(input_feature, is_training, l2_reg=self._l2_reg)
        block_outputs[block] = output
      elif layer == 'fm':
        input_feature = self.block_input(config, block_outputs)
        fm = FMLayer(config.fm, name=block)
        block_outputs[block] = fm(input_feature)
      elif layer == 'concat':
        input_feature = self.block_input(config, block_outputs)
        concat = Concatenate(config.concat)
        block_outputs[block] = concat(input_feature)
      elif layer == 'reshape':
        input_feature = self.block_input(config, block_outputs)
        block_outputs[block] = tf.reshape(input_feature, list(config.reshape.dims))
      elif layer == 'add':
        input_feature = self.block_input(config, block_outputs, output_list=True)
        block_outputs[block] = add_op(input_feature)
      elif layer == 'dot':
        input_feature = self.block_input(config, block_outputs)
        block_outputs[block] = dot_op(input_feature)
      elif layer == 'Lambda':
        input_feature = self.block_input(config, block_outputs)
        fn = eval(config.Lambda.expression)
        block_outputs[block] = fn(input_feature)
      elif layer == 'chain':
        input_feature = self.block_input(config, block_outputs)
        block_outputs[block] = op_chain(input_feature, config.chain.ops)
      else:
        raise NotImplementedError('Unsupported backbone layer:' + layer)

    temp = []
    for output in self._config.concat_blocks:
      if output in block_outputs:
        temp.append(block_outputs[output])
      else:
        raise ValueError('No output `%s` of backbone to be concat' % output)

    output = concat_inputs(temp, msg='backbone')
    if self._config.HasField('top_mlp'):
      no_act = self._config.top_mlp.last_layer_no_activation
      no_bn = self._config.top_mlp.last_layer_no_batch_norm
      final_dnn = dnn.DNN(
          self._config.top_mlp,
          self._l2_reg,
          name='backbone_top_mlp',
          is_training=is_training,
          last_layer_no_activation=no_act,
          last_layer_no_batch_norm=no_bn)
      output = final_dnn(output)
    return output

  def sequence_encoder(self, config, is_training):
    encodings = []
    for seq_input in config.inputs:
      encoding = self._model.get_sequence_encoding(seq_input, is_training)
      encodings.append(encoding)
    encoding = concat_inputs(encodings)
    conf = config.sequence_encoder
    if conf.HasField('mlp'):
      sequence_dnn = dnn.DNN(
          conf.mlp,
          self._l2_reg,
          name='%s_seq_dnn' % config.name,
          is_training=is_training)
      encoding = sequence_dnn(encoding)
    return encoding


def concat_inputs(inputs, axis=-1, msg=''):
  if len(inputs) > 1:
    if all(map(lambda x: type(x) == list, inputs)):
      # merge multiple lists into a list
      from functools import reduce
      return reduce(lambda x, y: x + y, inputs)

    if axis != -1:
      logging.info('concat inputs %s axis=%d' % (msg, axis))
    return tf.concat(inputs, axis=axis)

  if len(inputs) == 1:
    return inputs[0]
  raise ValueError('no inputs to be concat:' + msg)


def op_chain(inputs, ops):
  output = inputs
  for op in ops:
    op_name = op.WhichOneOf('Op')
    output = run_op(output, op_name, op, block='op_chain')
  return output


def run_op(inputs, op_name, config, block='', is_training=False, l2_reg=None):
  if op_name == 'periodic_embedding':
    num_emb = PeriodicEmbedding(config.periodic_embedding, scope=block)
    return num_emb(inputs)
  elif op_name == 'auto_dis_embedding':
    num_emb = AutoDisEmbedding(config.auto_dis_embedding, scope=block)
    return num_emb(inputs)
  elif op_name == 'highway':
    conf = config.highway
    highway_op_name = highway(
      inputs,
      conf.emb_size,
      activation=conf.activation,
      dropout=conf.dropout_rate,
      scope=block)
    return highway_op_name(inputs)
  elif op_name == 'mlp':
    mlp = dnn.DNN(
      config.mlp,
      l2_reg,
      name='%s_mlp' % block,
      is_training=is_training,
      last_layer_no_activation=config.mlp.last_layer_no_activation,
      last_layer_no_batch_norm=config.mlp.last_layer_no_batch_norm)
    return mlp(inputs)
  elif op_name == 'masknet':
    mask_net = MaskNet(config.masknet, name=block, reuse=tf.AUTO_REUSE)
    output = mask_net(inputs, is_training, l2_reg=l2_reg)
    return output
  elif op_name == 'senet':
    senet = SENet(config.senet, name=block)
    output = senet(inputs)
    return output
  elif op_name == 'fibinet':
    fibinet = FiBiNetLayer(config.fibinet, name=block)
    output = fibinet(inputs, is_training, l2_reg=l2_reg)
    return output
  elif op_name == 'fm':
    fm = FMLayer(config.fm, name=block)
    return fm(inputs)
  if op_name == 'Lambda':
    fn = eval(config.Lambda.expression)
    output = fn(inputs)
  elif op_name == 'concat':
    concat = Concatenate(config.concat)
    output = concat(inputs)
  elif op_name == 'reshape':
    output = tf.reshape(inputs, list(config.reshape.dims))
  elif op_name == 'add':
    output = add_op(inputs)
  elif op_name == 'dot':
    output = dot_op(inputs)
  else:
    raise NotImplementedError('Unsupported op:' + op_name)
  return output
