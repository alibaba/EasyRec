# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.utils.dag import DAG
from easy_rec.python.layers import dnn
from easy_rec.python.layers.common_layers import layer_norm, SENet, highway
from easy_rec.python.layers.numerical_embedding import PeriodicEmbedding, AutoDisEmbedding
from easy_rec.python.layers.fibinet import FiBiNetLayer
from easy_rec.python.layers.mask_net import MaskNet
from easy_rec.python.layers.fm import FMLayer

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class EnhancedInputLayer(object):
  def __init__(self, config, input_layer, feature_dict):
    if config.do_batch_norm and config.do_layer_norm:
      raise ValueError('can not do batch norm and layer norm for input layer at the same time')
    self._config = config
    self._input_layer = input_layer
    self._feature_dict = feature_dict

  def __call__(self, feature_group, is_training, *args, **kwargs):
    features, feature_list = self._input_layer(self._feature_dict, feature_group)
    num_features = len(feature_list)

    do_feature_dropout = 0.0 < self._config.feature_dropout_rate < 1.0
    if self._config.output_feature_list or do_feature_dropout:
      if self._config.do_layer_norm or self._config.do_batch_norm:
        for i in range(num_features):
          fea = feature_list[i]
          if self._config.do_batch_norm:
            fea = tf.layers.batch_normalization(fea, training=is_training)
          elif self._config.do_layer_norm:
            fea = layer_norm(fea)
          feature_list[i] = fea
    elif self._config.do_batch_norm:
      features = tf.layers.batch_normalization(features, training=is_training)
    elif self._config.do_layer_norm:
      features = layer_norm(features)

    if do_feature_dropout and is_training:
      keep_prob = 1.0 - self._config.feature_dropout_rate
      bern = tf.distributions.Bernoulli(probs=keep_prob)
      mask = bern.sample(num_features)
      for i in range(num_features):
        fea = tf.div(feature_list[i], keep_prob) * mask[i]
        feature_list[i] = fea
      features = tf.concat(feature_list, axis=-1)

    do_dropout = 0.0 < self._config.dropout_rate < 1.0
    if self._config.output_feature_list:
      if do_dropout:
        for i in range(num_features):
          fea = feature_list[i]
          fea = tf.layers.dropout(fea, self._config.dropout_rate, training=is_training)
          feature_list[i] = fea
      return feature_list
    if do_dropout:
      return tf.layers.dropout(features, self._config.dropout_rate, training=is_training)
    return features


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
    assert len(self._name_to_blocks) > 0, 'there must be more than one block in backbone'
    for block in config.blocks:
      assert len(block.inputs) > 0, 'there is no input for block: %s' % block.name
      for node in block.inputs:
        if node in self._name_to_blocks:
          self._dag.add_edge(node, block.name)

  def block_input(self, config, block_outputs):
    inputs = []
    for input_name in config.inputs:
      if input_name in block_outputs:
        input_feature = block_outputs[input_name]
      else:
        input_feature, _ = self._input_layer(self._features, input_name)
      inputs.append(input_feature)
    return concat_inputs(inputs, config.name)

  def __call__(self, is_training, *args, **kwargs):
    block_outputs = {}
    blocks = self._dag.topological_sort()
    logging.info("backbone topological order: " + ','.join(blocks))
    print("backbone topological order: " + ','.join(blocks))
    for block in blocks:
      config = self._name_to_blocks[block]
      layer = config.WhichOneof('layer')
      if layer == 'input_layer':
        assert len(config.inputs) == 1, 'only one input needed for input_layer: ' + block.name
        conf = config.input_layer
        input_layer = EnhancedInputLayer(conf, self._input_layer, self._features)
        output = input_layer(config.inputs[0], is_training)
        block_outputs[block] = output
      elif layer == 'periodic_embedding':
        input_feature = self.block_input(config, block_outputs)
        conf = config.periodic_embedding
        num_emb = PeriodicEmbedding(conf.embedding_dim, stddev=conf.coef_stddev, scope=block)
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
          is_training=is_training)
        block_outputs[block] = mlp(input_feature)
      elif layer == 'sequence_encoder':
        block_outputs[block] = self.sequence_encoder(config, is_training)
      elif layer == 'masknet':
        input_feature = self.block_input(config, block_outputs)
        mask_net = MaskNet(
          config.masknet,
          name=block,
          reuse=tf.AUTO_REUSE)
        output = mask_net(
          input_feature, is_training, l2_reg=self._l2_reg)
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
        fm = FMLayer()
        block_outputs[block] = fm(input_feature)
      else:
        raise NotImplementedError('Unsupported backbone layer:' + layer)

    temp = []
    for output in self._config.concat_blocks:
      if output in block_outputs:
        temp.append(block_outputs[output])
      else:
        raise ValueError('No output `%s` of backbone to be concat' % output)

    output = concat_inputs(temp)
    if self._config.HasField('top_mlp'):
      final_dnn = dnn.DNN(
        self._config.top_mlp,
        self._l2_reg,
        name='backbone_top_mlp',
        is_training=is_training)
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


def concat_inputs(inputs, msg=''):
  if len(inputs) > 1:
    if type(inputs[0]) == list:
      from functools import reduce
      return reduce(lambda x, y: x + y, inputs)
    return tf.concat(inputs, axis=-1)
  if len(inputs) == 1:
    return inputs[0]
  raise ValueError('no inputs to be concat:' + msg)


