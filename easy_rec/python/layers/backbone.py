# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers.common_layers import EnhancedInputLayer
from easy_rec.python.layers.keras import MLP
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.protos import backbone_pb2
from easy_rec.python.utils.dag import DAG
from easy_rec.python.utils.load_class import load_keras_layer
from google.protobuf import struct_pb2

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def block_input(config, block_outputs):
  inputs = []
  for input_node in config.inputs:
    input_name = input_node.name
    if input_name in block_outputs:
      input_feature = block_outputs[input_name]
    else:
      raise KeyError('input name `%s` does not exists' % input_name)
    if input_node.HasField('input_fn'):
      fn = eval(input_node.input_fn)
      input_feature = fn(input_feature)
    inputs.append(input_feature)

  if config.merge_inputs_into_list:
    output = inputs
  else:
    output = concat_inputs(inputs, config.input_concat_axis, config.name)

  if config.HasField('extra_input_fn'):
    fn = eval(config.extra_input_fn)
    output = fn(output)
  return output


class Backbone(object):
  """Configurable Backbone Network."""

  def __init__(self, config, features, input_layer, l2_reg=None):
    self._config = config
    self._features = features
    self._input_layer = input_layer
    self._l2_reg = l2_reg
    self._dag = DAG()
    self._name_to_blocks = {}
    self.loss_dict = {}
    input_feature_groups = set()
    for block in config.blocks:
      self._dag.add_node(block.name)
      self._name_to_blocks[block.name] = block
      layer = block.WhichOneof('layer')
      if layer == 'input_layer':
        if len(block.inputs) != 0:
          raise ValueError('no input allowed for input_layer: ' + block.name)
        input_name = block.name
        if not input_layer.has_group(input_name):
          raise KeyError(
              'input_layer\'s name must be one of feature group, invalid: ' +
              input_name)
        if input_name in input_feature_groups:
          raise ValueError('input `%s` already exists in other block' %
                           input_name)
        else:
          input_feature_groups.add(input_name)

    num_groups = len(input_feature_groups)
    num_blocks = len(self._name_to_blocks) - num_groups
    assert num_blocks > 0, 'there must be at least one block in backbone'

    for block in config.blocks:
      layer = block.WhichOneof('layer')
      if layer == 'input_layer':
        continue
      if block.name in input_feature_groups:
        raise KeyError('block name can not be one of feature groups:' +
                       block.name)
      assert len(block.inputs) > 0, 'no input for block: %s' % block.name

      for input_node in block.inputs:
        input_name = input_node.name
        if input_name in self._name_to_blocks:
          assert input_name != block.name, 'input name can not equal to block name:' + input_name
          self._dag.add_edge(input_name, block.name)
        elif input_name not in input_feature_groups:
          if input_layer.has_group(input_name):
            logging.info('adding an input_layer block: ' + input_name)
            new_block = backbone_pb2.Block()
            new_block.name = input_name
            new_block.input_layer.CopyFrom(backbone_pb2.InputLayer())
            self._name_to_blocks[input_name] = new_block
            self._dag.add_node(input_name)
            self._dag.add_edge(input_name, block.name)
            input_feature_groups.add(block.name)
          else:
            raise KeyError(
                'invalid input name `%s`, must be the name of either a feature group or an another block'
                % input_name)
    num_groups = len(input_feature_groups)
    assert num_groups > 0, 'there must be at least one input layer'

  def __call__(self, is_training, **kwargs):
    block_outputs = {}
    blocks = self._dag.topological_sort()
    logging.info('backbone topological order: ' + ','.join(blocks))
    print('backbone topological order: ' + ','.join(blocks))
    for block in blocks:
      config = self._name_to_blocks[block]
      layer = config.WhichOneof('layer')
      if layer is None:  # identity layer
        block_outputs[block] = block_input(config, block_outputs)
      elif layer == 'input_layer':
        conf = config.input_layer
        input_fn = EnhancedInputLayer(conf, self._input_layer, self._features)
        output = input_fn(block, is_training)
        block_outputs[block] = output
      elif layer == 'sequential':
        inputs = block_input(config, block_outputs)
        layers = config.sequential.layers
        output = self.call_sequential_layers(inputs, layers, block, is_training)
        block_outputs[block] = output
      else:
        inputs = block_input(config, block_outputs)
        block_outputs[block] = self.call_layer(inputs, config, block, is_training)

    temp = []
    for output in self._config.concat_blocks:
      if output in block_outputs:
        temp.append(block_outputs[output])
      else:
        raise ValueError('No output `%s` of backbone to be concat' % output)
    output = concat_inputs(temp, msg='backbone')

    if self._config.HasField('top_mlp'):
      params = Parameter.make_from_pb(self._config.top_mlp)
      params.l2_regularizer = self._l2_reg
      final_mlp = MLP(params, name='backbone_top_mlp')
      output = final_mlp(output, training=is_training)
    return output

  def call_keras_layer(self, layer_conf, inputs, name, training):
    layer_cls, customize = load_keras_layer(layer_conf.class_name)
    if layer_cls is None:
      raise ValueError('Invalid keras layer class name: ' +
                       layer_conf.class_name)

    param_type = layer_conf.WhichOneof('params')
    if customize:
      if param_type is None or param_type == 'st_params':
        params = Parameter(layer_conf.st_params, True, l2_reg=self._l2_reg)
      else:
        pb_params = getattr(layer_conf, param_type)
        params = Parameter(pb_params, False, l2_reg=self._l2_reg)
      layer = layer_cls(params, name=name)
      kwargs = {'loss_dict': self.loss_dict}
      return layer(inputs, training=training, **kwargs)
    else:  # internal keras layer
      if param_type is None:
        layer = layer_cls(name=name)
      else:
        assert param_type == 'st_params', 'internal keras layer only support st_params'
        kwargs = convert_to_dict(layer_conf.st_params)
        layer = layer_cls(name=name, **kwargs)
      return layer(inputs, training=training)

  def call_sequential_layers(self, inputs, layers, name, training):
   output = inputs
   for layer in layers:
     output = self.call_layer(output, layer, name, training)
   return output

  def call_layer(self, inputs, config, name, training):
    layer_name = config.WhichOneof('layer')
    if layer_name == 'keras_layer':
      return self.call_keras_layer(config.keras_layer, inputs, name, training)
    if layer_name == 'lambda':
      conf = getattr(config, 'lambda')
      fn = eval(conf.expression)
      return fn(inputs)
    raise NotImplementedError('Unsupported backbone layer:' + layer_name)


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


def format_value(value):
  value_type = type(value)
  if value_type in (unicode, str):
    return str(value)
  if value_type == float:
    int_v = int(value)
    return int_v if int_v == value else value
  if value_type == struct_pb2.ListValue:
    return map(format_value, value)
  if value_type == struct_pb2.Struct:
    return convert_to_dict(value)
  return value


def convert_to_dict(struct):
  kwargs = {}
  for key, value in struct.items():
    kwargs[str(key)] = format_value(value)
  return kwargs

