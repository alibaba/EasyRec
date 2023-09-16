# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import six
import tensorflow as tf
from google.protobuf import struct_pb2

from easy_rec.python.layers.common_layers import EnhancedInputLayer
from easy_rec.python.layers.keras import MLP
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.protos import backbone_pb2
from easy_rec.python.utils.dag import DAG
from easy_rec.python.utils.load_class import load_keras_layer

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class Package(object):
  """A sub DAG of tf ops for reuse."""
  __packages = {}

  def __init__(self, config, features, input_layer, l2_reg=None):
    self._config = config
    self._features = features
    self._input_layer = input_layer
    self._l2_reg = l2_reg
    self._dag = DAG()
    self._name_to_blocks = {}
    self.loss_dict = {}
    self._name_to_layer = {}
    reuse = None if config.name == 'backbone' else tf.AUTO_REUSE
    input_feature_groups = set()
    for block in config.blocks:
      if len(block.inputs) == 0:
        raise ValueError('block takes at least one input: %s' % block.name)
      self._dag.add_node(block.name)
      self._name_to_blocks[block.name] = block
      layer = block.WhichOneof('layer')
      if layer == 'input_layer':
        if len(block.inputs) != 1:
          raise ValueError('input layer `%s` takes only one input' % block.name)
        one_input = block.inputs[0]
        name = one_input.WhichOneof('name')
        if name != 'feature_group_name':
          raise KeyError(
              '`feature_group_name` should be set for input layer: ' +
              block.name)
        input_name = one_input.feature_group_name
        if not input_layer.has_group(input_name):
          raise KeyError('invalid feature group name: ' + input_name)
        if input_name in input_feature_groups:
          logging.warning('input `%s` already exists in other block' %
                          input_name)
        input_feature_groups.add(input_name)
      else:
        self.define_layers(layer, block, block.name, reuse)

      # sequential layers
      for i, layer_cnf in enumerate(block.layers):
        layer = layer_cnf.WhichOneof('layer')
        name_i = '%s_l%d' % (block.name, i)
        self.define_layers(layer, layer_cnf, name_i, reuse)

    num_groups = len(input_feature_groups)
    num_blocks = len(self._name_to_blocks) - num_groups
    assert num_blocks > 0, 'there must be at least one block in backbone'

    num_pkg_input = 0
    for block in config.blocks:
      layer = block.WhichOneof('layer')
      if layer == 'input_layer':
        continue
      if block.name in input_feature_groups:
        raise KeyError('block name can not be one of feature groups:' +
                       block.name)
      for input_node in block.inputs:
        input_type = input_node.WhichOneof('name')
        if input_type == 'package_name':
          num_pkg_input += 1
          continue
        input_name = getattr(input_node, input_type)
        if input_name in self._name_to_blocks:
          assert input_name != block.name, 'input name can not equal to block name:' + input_name
          self._dag.add_edge(input_name, block.name)
        elif input_name not in input_feature_groups:
          if input_layer.has_group(input_name):
            logging.info('adding an input_layer block: ' + input_name)
            new_block = backbone_pb2.Block()
            new_block.name = input_name
            input_cfg = backbone_pb2.Input()
            input_cfg.feature_group_name = input_name
            new_block.inputs.append(input_cfg)
            new_block.input_layer.CopyFrom(backbone_pb2.InputLayer())
            self._name_to_blocks[input_name] = new_block
            self._dag.add_node(input_name)
            self._dag.add_edge(input_name, block.name)
            input_feature_groups.add(input_name)
          else:
            raise KeyError(
                'invalid input name `%s`, must be the name of either a feature group or an another block'
                % input_name)
    num_groups = len(input_feature_groups)
    assert num_pkg_input > 0 or num_groups > 0, 'there must be at least one input layer/feature group'

    if len(config.concat_blocks) == 0:
      leaf = self._dag.all_leaves()
      logging.warning(
          '%s has no `concat_blocks`, try to use all leaf blocks: %s' %
          (config.name, ','.join(leaf)))
      self._config.concat_blocks.extend(leaf)

    Package.__packages[self._config.name] = self

  def define_layers(self, layer, layer_cnf, name, reuse):
    if layer == 'keras_layer':
      layer_obj = self.load_keras_layer(layer_cnf.keras_layer, name, reuse)
      self._name_to_layer[name] = layer_obj
    elif layer == 'recurrent':
      for i in range(layer_cnf.recurrent.num_steps):
        name_i = '%s_%d' % (name, i)
        layer_obj = self.load_keras_layer(layer_cnf.recurrent.keras_layer,
                                          name_i, reuse)
        self._name_to_layer[name_i] = layer_obj
    elif layer == 'repeat':
      for i in range(layer_cnf.repeat.num_repeat):
        name_i = '%s_%d' % (name, i)
        layer_obj = self.load_keras_layer(layer_cnf.repeat.keras_layer, name_i,
                                          reuse)
        self._name_to_layer[name_i] = layer_obj

  def block_input(self, config, block_outputs, training=None):
    inputs = []
    for input_node in config.inputs:
      input_type = input_node.WhichOneof('name')
      input_name = getattr(input_node, input_type)
      if input_type == 'package_name':
        if input_name not in Package.__packages:
          raise KeyError('package name `%s` does not exists' % input_name)
        package = Package.__packages[input_name]
        input_feature = package(training)
        if len(package.loss_dict) > 0:
          self.loss_dict.update(package.loss_dict)
      elif input_name in block_outputs:
        input_feature = block_outputs[input_name]
      else:
        raise KeyError('input name `%s` does not exists' % input_name)

      if input_node.HasField('input_slice'):
        fn = eval('lambda x: x' + input_node.input_slice.strip())
        input_feature = fn(input_feature)
      if input_node.HasField('input_fn'):
        fn = eval(input_node.input_fn)
        input_feature = fn(input_feature)
      inputs.append(input_feature)

    if config.merge_inputs_into_list:
      output = inputs
    else:
      output = merge_inputs(inputs, config.input_concat_axis, config.name)

    if config.HasField('extra_input_fn'):
      fn = eval(config.extra_input_fn)
      output = fn(output)
    return output

  def __call__(self, is_training, **kwargs):
    with tf.variable_scope(self._config.name, reuse=tf.AUTO_REUSE):
      return self.call(is_training)

  def call(self, is_training):
    block_outputs = {}
    blocks = self._dag.topological_sort()
    logging.info(self._config.name + ' topological order: ' + ','.join(blocks))
    print(self._config.name + ' topological order: ' + ','.join(blocks))
    for block in blocks:
      config = self._name_to_blocks[block]
      if config.layers:  # sequential layers
        logging.info('call sequential %d layers' % len(config.layers))
        output = self.block_input(config, block_outputs, is_training)
        for i, layer in enumerate(config.layers):
          name_i = '%s_l%d' % (block, i)
          output = self.call_layer(output, layer, name_i, is_training)
        block_outputs[block] = output
        continue
      # just one of layer
      layer = config.WhichOneof('layer')
      if layer is None:  # identity layer
        block_outputs[block] = self.block_input(config, block_outputs,
                                                is_training)
      elif layer == 'input_layer':
        conf = config.input_layer
        input_fn = EnhancedInputLayer(conf, self._input_layer, self._features)
        feature_group = config.inputs[0].feature_group_name
        output = input_fn(feature_group, is_training)
        block_outputs[block] = output
      else:
        inputs = self.block_input(config, block_outputs, is_training)
        output = self.call_layer(inputs, config, block, is_training)
        block_outputs[block] = output

    outputs = []
    for output in self._config.concat_blocks:
      if output in block_outputs:
        temp = block_outputs[output]
        if type(temp) in (tuple, list):
          outputs.extend(temp)
        else:
          outputs.append(temp)
      else:
        raise ValueError('No output `%s` of backbone to be concat' % output)
    output = merge_inputs(outputs, msg='backbone')
    return output

  def load_keras_layer(self, layer_conf, name, reuse=None):
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
      layer = layer_cls(params, name=name, reuse=reuse)
      return layer
    elif param_type is None:  # internal keras layer
      layer = layer_cls(name=name)
      return layer
    else:
      assert param_type == 'st_params', 'internal keras layer only support st_params'
      try:
        kwargs = convert_to_dict(layer_conf.st_params)
        logging.info('call %s layer with params %r' %
                     (layer_conf.class_name, kwargs))
        layer = layer_cls(name=name, **kwargs)
      except TypeError as e:
        logging.warning(e)
        args = map(format_value, layer_conf.st_params.values())
        logging.info('try to call %s layer with params %r' %
                     (layer_conf.class_name, args))
        layer = layer_cls(*args, name=name)
      return layer

  def call_keras_layer(self, inputs, name, training):
    """Call predefined Keras Layer, which can be reused."""
    layer = self._name_to_layer[name]
    kwargs = {'loss_dict': self.loss_dict}
    try:
      return layer(inputs, training=training, **kwargs)
    except TypeError:
      return layer(inputs)

  def call_layer(self, inputs, config, name, training):
    layer_name = config.WhichOneof('layer')
    if layer_name == 'keras_layer':
      return self.call_keras_layer(inputs, name, training)
    if layer_name == 'lambda':
      conf = getattr(config, 'lambda')
      fn = eval(conf.expression)
      return fn(inputs)
    if layer_name == 'repeat':
      conf = config.repeat
      n_loop = conf.num_repeat
      outputs = []
      for i in range(n_loop):
        name_i = '%s_%d' % (name, i)
        output = self.call_keras_layer(inputs, name_i, training)
        outputs.append(output)
      if len(outputs) == 1:
        return outputs[0]
      if conf.HasField('output_concat_axis'):
        return tf.concat(outputs, conf.output_concat_axis)
      return outputs
    if layer_name == 'recurrent':
      conf = config.recurrent
      fixed_input_index = -1
      if conf.HasField('fixed_input_index'):
        fixed_input_index = conf.fixed_input_index
      if fixed_input_index >= 0:
        assert type(inputs) in (tuple, list), '%s inputs must be a list'
      output = inputs
      for i in range(conf.num_steps):
        name_i = '%s_%d' % (name, i)
        output_i = self.call_keras_layer(output, name_i, training)
        if fixed_input_index >= 0:
          j = 0
          for idx in range(len(output)):
            if idx == fixed_input_index:
              continue
            if type(output_i) in (tuple, list):
              output[idx] = output_i[j]
            else:
              output[idx] = output_i
            j += 1
        else:
          output = output_i
      if fixed_input_index >= 0:
        del output[fixed_input_index]
        if len(output) == 1:
          return output[0]
        return output
      return output

    raise NotImplementedError('Unsupported backbone layer:' + layer_name)


class Backbone(object):
  """Configurable Backbone Network."""

  def __init__(self, config, features, input_layer, l2_reg=None):
    self._config = config
    self._l2_reg = l2_reg
    self.loss_dict = {}
    for pkg in config.packages:
      Package(pkg, features, input_layer, l2_reg)

    main_pkg = backbone_pb2.BlockPackage()
    main_pkg.name = 'backbone'
    main_pkg.blocks.MergeFrom(config.blocks)
    main_pkg.concat_blocks.extend(config.concat_blocks)
    self._main_pkg = Package(main_pkg, features, input_layer, l2_reg)

  def __call__(self, is_training, **kwargs):
    output = self._main_pkg(is_training, **kwargs)
    if len(self._main_pkg.loss_dict) > 0:
      self.loss_dict = self._main_pkg.loss_dict

    if self._config.HasField('top_mlp'):
      params = Parameter.make_from_pb(self._config.top_mlp)
      params.l2_regularizer = self._l2_reg
      final_mlp = MLP(params, name='backbone_top_mlp')
      output = final_mlp(output, training=is_training)
    return output

  @classmethod
  def wide_embed_dim(cls, config):
    wide_embed_dim = None
    for pkg in config.packages:
      wide_embed_dim = get_wide_embed_dim(pkg.blocks, wide_embed_dim)
    return get_wide_embed_dim(config.blocks, wide_embed_dim)


def get_wide_embed_dim(blocks, wide_embed_dim=None):
  for block in blocks:
    layer = block.WhichOneof('layer')
    if layer == 'input_layer':
      if block.input_layer.HasField('wide_output_dim'):
        wide_dim = block.input_layer.wide_output_dim
        if wide_embed_dim:
          assert wide_embed_dim == wide_dim, 'wide_output_dim must be consistent'
        else:
          wide_embed_dim = wide_dim
  return wide_embed_dim


def merge_inputs(inputs, axis=-1, msg=''):
  if len(inputs) == 0:
    raise ValueError('no inputs to be concat:' + msg)
  if len(inputs) == 1:
    return inputs[0]

  from functools import reduce
  if all(map(lambda x: type(x) == list, inputs)):
    # merge multiple lists into a list
    return reduce(lambda x, y: x + y, inputs)

  if any(map(lambda x: type(x) == list, inputs)):
    logging.warning('%s: try to merge inputs into list' % msg)
    return reduce(lambda x, y: x + y,
                  [e if type(e) == list else [e] for e in inputs])

  if axis != -1:
    logging.info('concat inputs %s axis=%d' % (msg, axis))
  return tf.concat(inputs, axis=axis)


def format_value(value):
  value_type = type(value)
  if value_type == six.text_type:
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
