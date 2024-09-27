# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import six
import tensorflow as tf
from google.protobuf import struct_pb2

from easy_rec.python.layers.common_layers import EnhancedInputLayer
from easy_rec.python.layers.keras import MLP
from easy_rec.python.layers.keras import EmbeddingLayer
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.protos import backbone_pb2
from easy_rec.python.utils.dag import DAG
from easy_rec.python.utils.load_class import load_keras_layer
from easy_rec.python.utils.tf_utils import add_elements_to_collection

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class Package(object):
  """A sub DAG of tf ops for reuse."""
  __packages = {}

  @staticmethod
  def has_backbone_block(name):
    if 'backbone' not in Package.__packages:
      return False
    backbone = Package.__packages['backbone']
    return backbone.has_block(name)

  @staticmethod
  def backbone_block_outputs(name):
    if 'backbone' not in Package.__packages:
      return None
    backbone = Package.__packages['backbone']
    return backbone.block_outputs(name)

  def __init__(self, config, features, input_layer, l2_reg=None):
    self._config = config
    self._features = features
    self._input_layer = input_layer
    self._l2_reg = l2_reg
    self._dag = DAG()
    self._name_to_blocks = {}
    self._name_to_layer = {}
    self.reset_input_config(None)
    self._block_outputs = {}
    self._package_input = None
    self._feature_group_inputs = {}
    reuse = None if config.name == 'backbone' else tf.AUTO_REUSE
    input_feature_groups = self._feature_group_inputs

    for block in config.blocks:
      if len(block.inputs) == 0:
        raise ValueError('block takes at least one input: %s' % block.name)
      self._dag.add_node(block.name)
      self._name_to_blocks[block.name] = block
      layer = block.WhichOneof('layer')
      if layer in {'input_layer', 'raw_input', 'embedding_layer'}:
        if len(block.inputs) != 1:
          raise ValueError('input layer `%s` takes only one input' % block.name)
        one_input = block.inputs[0]
        name = one_input.WhichOneof('name')
        if name != 'feature_group_name':
          raise KeyError(
              '`feature_group_name` should be set for input layer: ' +
              block.name)
        group = one_input.feature_group_name
        if not input_layer.has_group(group):
          raise KeyError('invalid feature group name: ' + group)
        if group in input_feature_groups:
          if layer == input_layer:
            logging.warning('input `%s` already exists in other block' % group)
          elif layer == 'raw_input':
            input_fn = input_feature_groups[group]
            self._name_to_layer[block.name] = input_fn
          elif layer == 'embedding_layer':
            inputs, vocab, weights = input_feature_groups[group]
            block.embedding_layer.vocab_size = vocab
            params = Parameter.make_from_pb(block.embedding_layer)
            input_fn = EmbeddingLayer(params, block.name)
            self._name_to_layer[block.name] = input_fn
        else:
          if layer == 'input_layer':
            input_fn = EnhancedInputLayer(self._input_layer, self._features,
                                          group, reuse)
            input_feature_groups[group] = input_fn
          elif layer == 'raw_input':
            input_fn = self._input_layer.get_raw_features(self._features, group)
            input_feature_groups[group] = input_fn
          else:  # embedding_layer
            inputs, vocab, weights = self._input_layer.get_bucketized_features(
                self._features, group)
            block.embedding_layer.vocab_size = vocab
            params = Parameter.make_from_pb(block.embedding_layer)
            input_fn = EmbeddingLayer(params, block.name)
            input_feature_groups[group] = (inputs, vocab, weights)
            logging.info('add an embedding layer %s with vocab size %d',
                         block.name, vocab)
          self._name_to_layer[block.name] = input_fn
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
      if layer in {'input_layer', 'raw_input', 'embedding_layer'}:
        continue
      name = block.name
      if name in input_feature_groups:
        raise KeyError('block name can not be one of feature groups:' + name)
      for input_node in block.inputs:
        input_type = input_node.WhichOneof('name')
        input_name = getattr(input_node, input_type)
        if input_type == 'use_package_input':
          assert input_name, 'use_package_input can not set false'
          num_pkg_input += 1
          continue
        if input_type == 'package_name':
          num_pkg_input += 1
          self._dag.add_node_if_not_exists(input_name)
          self._dag.add_edge(input_name, name)
          if input_node.HasField('package_input'):
            pkg_input_name = input_node.package_input
            self._dag.add_node_if_not_exists(pkg_input_name)
            self._dag.add_edge(pkg_input_name, input_name)
          continue
        iname = input_name
        if iname in self._name_to_blocks:
          assert iname != name, 'input name can not equal to block name:' + iname
          self._dag.add_edge(iname, name)
        else:
          is_fea_group = input_type == 'feature_group_name'
          if is_fea_group and input_layer.has_group(iname):
            logging.info('adding an input_layer block: ' + iname)
            new_block = backbone_pb2.Block()
            new_block.name = iname
            input_cfg = backbone_pb2.Input()
            input_cfg.feature_group_name = iname
            new_block.inputs.append(input_cfg)
            new_block.input_layer.CopyFrom(backbone_pb2.InputLayer())
            self._name_to_blocks[iname] = new_block
            self._dag.add_node(iname)
            self._dag.add_edge(iname, name)
            if iname in input_feature_groups:
              fn = input_feature_groups[iname]
            else:
              fn = EnhancedInputLayer(self._input_layer, self._features, iname)
              input_feature_groups[iname] = fn
            self._name_to_layer[iname] = fn
          elif Package.has_backbone_block(iname):
            backbone = Package.__packages['backbone']
            backbone._dag.add_node_if_not_exists(self._config.name)
            backbone._dag.add_edge(iname, self._config.name)
            num_pkg_input += 1
          else:
            raise KeyError(
                'invalid input name `%s`, must be the name of either a feature group or an another block'
                % iname)
    num_groups = len(input_feature_groups)
    assert num_pkg_input > 0 or num_groups > 0, 'there must be at least one input layer/feature group'

    if len(config.concat_blocks) == 0 and len(config.output_blocks) == 0:
      leaf = self._dag.all_leaves()
      logging.warning(
          '%s has no `concat_blocks` or `output_blocks`, try to concat all leaf blocks: %s'
          % (config.name, ','.join(leaf)))
      self._config.concat_blocks.extend(leaf)

    Package.__packages[self._config.name] = self
    logging.info('%s layers: %s' %
                 (config.name, ','.join(self._name_to_layer.keys())))

  def define_layers(self, layer, layer_cnf, name, reuse):
    if layer == 'keras_layer':
      layer_obj = self.load_keras_layer(layer_cnf.keras_layer, name, reuse)
      self._name_to_layer[name] = layer_obj
    elif layer == 'recurrent':
      keras_layer = layer_cnf.recurrent.keras_layer
      for i in range(layer_cnf.recurrent.num_steps):
        name_i = '%s_%d' % (name, i)
        layer_obj = self.load_keras_layer(keras_layer, name_i, reuse)
        self._name_to_layer[name_i] = layer_obj
    elif layer == 'repeat':
      keras_layer = layer_cnf.repeat.keras_layer
      for i in range(layer_cnf.repeat.num_repeat):
        name_i = '%s_%d' % (name, i)
        layer_obj = self.load_keras_layer(keras_layer, name_i, reuse)
        self._name_to_layer[name_i] = layer_obj

  def reset_input_config(self, config):
    self.input_config = config

  def set_package_input(self, pkg_input):
    self._package_input = pkg_input

  def has_block(self, name):
    return name in self._name_to_blocks

  def block_outputs(self, name):
    return self._block_outputs.get(name, None)

  def block_input(self, config, block_outputs, training=None, **kwargs):
    inputs = []
    for input_node in config.inputs:
      input_type = input_node.WhichOneof('name')
      input_name = getattr(input_node, input_type)
      if input_type == 'use_package_input':
        input_feature = self._package_input
        input_name = 'package_input'
      elif input_type == 'package_name':
        if input_name not in Package.__packages:
          raise KeyError('package name `%s` does not exists' % input_name)
        package = Package.__packages[input_name]
        if input_node.HasField('reset_input'):
          package.reset_input_config(input_node.reset_input)
        if input_node.HasField('package_input'):
          pkg_input_name = input_node.package_input
          if pkg_input_name in block_outputs:
            pkg_input = block_outputs[pkg_input_name]
          else:
            if pkg_input_name not in Package.__packages:
              raise KeyError('package name `%s` does not exists' %
                             pkg_input_name)
            inner_package = Package.__packages[pkg_input_name]
            pkg_input = inner_package(training)
          if input_node.HasField('package_input_fn'):
            fn = eval(input_node.package_input_fn)
            pkg_input = fn(pkg_input)
          package.set_package_input(pkg_input)
        input_feature = package(training, **kwargs)
      elif input_name in block_outputs:
        input_feature = block_outputs[input_name]
      else:
        input_feature = Package.backbone_block_outputs(input_name)

      if input_feature is None:
        raise KeyError('input name `%s` does not exists' % input_name)

      if input_node.ignore_input:
        continue
      if input_node.HasField('input_slice'):
        fn = eval('lambda x: x' + input_node.input_slice.strip())
        input_feature = fn(input_feature)
      if input_node.HasField('input_fn'):
        with tf.name_scope(config.name):
          fn = eval(input_node.input_fn)
          input_feature = fn(input_feature)
      inputs.append(input_feature)

    if config.merge_inputs_into_list:
      output = inputs
    else:
      try:
        output = merge_inputs(inputs, config.input_concat_axis, config.name)
      except ValueError as e:
        msg = getattr(e, 'message', str(e))
        logging.error('merge inputs of block %s failed: %s', config.name, msg)
        raise e

    if config.HasField('extra_input_fn'):
      fn = eval(config.extra_input_fn)
      output = fn(output)
    return output

  def __call__(self, is_training, **kwargs):
    with tf.name_scope(self._config.name):
      return self.call(is_training, **kwargs)

  def call(self, is_training, **kwargs):
    block_outputs = {}
    self._block_outputs = block_outputs  # reset
    blocks = self._dag.topological_sort()
    logging.info(self._config.name + ' topological order: ' + ','.join(blocks))
    for block in blocks:
      if block not in self._name_to_blocks:
        assert block in Package.__packages, 'invalid block: ' + block
        continue
      config = self._name_to_blocks[block]
      if config.layers:  # sequential layers
        logging.info('call sequential %d layers' % len(config.layers))
        output = self.block_input(config, block_outputs, is_training, **kwargs)
        for i, layer in enumerate(config.layers):
          name_i = '%s_l%d' % (block, i)
          output = self.call_layer(output, layer, name_i, is_training, **kwargs)
        block_outputs[block] = output
        continue
      # just one of layer
      layer = config.WhichOneof('layer')
      if layer is None:  # identity layer
        output = self.block_input(config, block_outputs, is_training, **kwargs)
        block_outputs[block] = output
      elif layer == 'raw_input':
        block_outputs[block] = self._name_to_layer[block]
      elif layer == 'input_layer':
        input_fn = self._name_to_layer[block]
        input_config = config.input_layer
        if self.input_config is not None:
          input_config = self.input_config
          input_fn.reset(input_config, is_training)
        block_outputs[block] = input_fn(input_config, is_training)
      elif layer == 'embedding_layer':
        input_fn = self._name_to_layer[block]
        feature_group = config.inputs[0].feature_group_name
        inputs, _, weights = self._feature_group_inputs[feature_group]
        block_outputs[block] = input_fn([inputs, weights], is_training)
      else:
        with tf.name_scope(block + '_input'):
          inputs = self.block_input(config, block_outputs, is_training,
                                    **kwargs)
        output = self.call_layer(inputs, config, block, is_training, **kwargs)
        block_outputs[block] = output

    outputs = []
    for output in self._config.output_blocks:
      if output in block_outputs:
        temp = block_outputs[output]
        outputs.append(temp)
      else:
        raise ValueError('No output `%s` of backbone to be concat' % output)
    if outputs:
      return outputs

    for output in self._config.concat_blocks:
      if output in block_outputs:
        temp = block_outputs[output]
        outputs.append(temp)
      else:
        raise ValueError('No output `%s` of backbone to be concat' % output)
    try:
      output = merge_inputs(outputs, msg='backbone')
    except ValueError as e:
      msg = getattr(e, 'message', str(e))
      logging.error("merge backbone's output failed: %s", msg)
      raise e
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

      has_reuse = True
      try:
        from funcsigs import signature
        sig = signature(layer_cls.__init__)
        has_reuse = 'reuse' in sig.parameters.keys()
      except ImportError:
        try:
          from sklearn.externals.funcsigs import signature
          sig = signature(layer_cls.__init__)
          has_reuse = 'reuse' in sig.parameters.keys()
        except ImportError:
          logging.warning('import funcsigs failed')

      if has_reuse:
        layer = layer_cls(params, name=name, reuse=reuse)
      else:
        layer = layer_cls(params, name=name)
      return layer, customize
    elif param_type is None:  # internal keras layer
      layer = layer_cls(name=name)
      return layer, customize
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
      return layer, customize

  def call_keras_layer(self, inputs, name, training, **kwargs):
    """Call predefined Keras Layer, which can be reused."""
    layer, customize = self._name_to_layer[name]
    cls = layer.__class__.__name__
    if customize:
      try:
        output = layer(inputs, training=training, **kwargs)
      except Exception as e:
        msg = getattr(e, 'message', str(e))
        logging.error('call keras layer %s (%s) failed: %s' % (name, cls, msg))
        raise e
    else:
      try:
        output = layer(inputs, training=training)
        if cls == 'BatchNormalization':
          add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
      except TypeError:
        output = layer(inputs)
    return output

  def call_layer(self, inputs, config, name, training, **kwargs):
    layer_name = config.WhichOneof('layer')
    if layer_name == 'keras_layer':
      return self.call_keras_layer(inputs, name, training, **kwargs)
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
        ly_inputs = inputs
        if conf.HasField('input_slice'):
          fn = eval('lambda x, i: x' + conf.input_slice.strip())
          ly_inputs = fn(ly_inputs, i)
        if conf.HasField('input_fn'):
          with tf.name_scope(config.name):
            fn = eval(conf.input_fn)
            ly_inputs = fn(ly_inputs, i)
        output = self.call_keras_layer(ly_inputs, name_i, training, **kwargs)
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
        output_i = self.call_keras_layer(output, name_i, training, **kwargs)
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
    main_pkg = backbone_pb2.BlockPackage()
    main_pkg.name = 'backbone'
    main_pkg.blocks.MergeFrom(config.blocks)
    if config.concat_blocks:
      main_pkg.concat_blocks.extend(config.concat_blocks)
    if config.output_blocks:
      main_pkg.output_blocks.extend(config.output_blocks)
    self._main_pkg = Package(main_pkg, features, input_layer, l2_reg)
    for pkg in config.packages:
      Package(pkg, features, input_layer, l2_reg)

  def __call__(self, is_training, **kwargs):
    output = self._main_pkg(is_training, **kwargs)

    if self._config.HasField('top_mlp'):
      params = Parameter.make_from_pb(self._config.top_mlp)
      params.l2_regularizer = self._l2_reg
      final_mlp = MLP(params, name='backbone_top_mlp')
      if type(output) in (list, tuple):
        output = tf.concat(output, axis=-1)
      output = final_mlp(output, training=is_training, **kwargs)
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
