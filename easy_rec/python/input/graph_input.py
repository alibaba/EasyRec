# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from easy_rec.python.core import sampler
from easy_rec.python.input.input import Input
from easy_rec.python.utils import graph_utils

try:
  import graphlearn as gl
except ImportError:
  logging.error(
      'GraphLearn is not installed. You can install it by "pip install https://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/graphlearn-0.7-cp27-cp27mu-linux_x86_64.whl"'  # noqa: E501
  )
  sys.exit(1)

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class GraphInput(Input):

  node_type = 'item'
  edge_type = 'relation'

  graph = None

  def __init__(self,
               data_config,
               feature_configs,
               input_path,
               task_index=0,
               task_num=1):
    super(GraphInput, self).__init__(data_config, feature_configs, input_path,
                                     task_index, task_num)
    self._model = None
    self._build_field_types()

    self._walk_len = self._data_config.graph_config.random_walk_len
    self._window_size = self._data_config.graph_config.window_size
    self._negative_num = self._data_config.graph_config.negative_num
    logging.info('walk_len=%d window_size=%d negative_num=%d' %
                 (self._walk_len, self._window_size, self._negative_num))

    # build item co-occurance graph, the weight is co-occurance frequency
    if input_path:
      if GraphInput.graph is None:
        GraphInput.graph = gl.Graph().node(
            str(input_path.node_inputs[0]),
            node_type=GraphInput.node_type,
            decoder=gl.Decoder(
                attr_types=self._attr_gl_types,
                weighted=True,
                attr_delimiter=',')).edge(
                    str(input_path.edge_inputs[0]),
                    edge_type=(GraphInput.node_type, GraphInput.node_type,
                               GraphInput.edge_type),
                    decoder=gl.Decoder(weighted=True),
                    directed=data_config.graph_config.directed)
        graph_utils.graph_init(GraphInput.graph,
                               os.environ.get('TF_CONFIG', None))
    if GraphInput.graph is not None:
      self._neg_sampler = GraphInput.graph.negative_sampler(
          GraphInput.node_type,
          expand_factor=self._negative_num,
          strategy='node_weight')

  def _build_field_types(self):
    """Build field types for item features."""
    self._attr_names = []
    self._attr_types = []
    self._attr_gl_types = []
    self._attr_np_types = []
    self._attr_tf_types = []
    for field_name, field_type in zip(self._input_fields,
                                      self._input_field_types):
      self._attr_names.append(field_name)
      self._attr_types.append(field_type)
      self._attr_gl_types.append(sampler._get_gl_type(field_type))
      self._attr_np_types.append(sampler._get_np_type(field_type))
      self._attr_tf_types.append(sampler._get_tf_type(field_type))

  def _parse_nodes(self, nodes):
    features = []
    int_idx = 0
    float_idx = 0
    string_idx = 0

    for attr_gl_type, attr_np_type in zip(self._attr_gl_types,
                                          self._attr_np_types):
      if attr_gl_type == 'int':
        if len(nodes.shape) == 1:
          feature = nodes.int_attrs[:, int_idx]
        elif len(nodes.shape) == 2:
          feature = nodes.int_attrs[:, :, int_idx]
        int_idx += 1
      elif attr_gl_type == 'float':
        if len(nodes.shape) == 1:
          feature = nodes.float_attrs[:, float_idx]
        elif len(nodes.shape) == 2:
          feature = nodes.float_attrs[:, :, float_idx]
        float_idx += 1
      elif attr_gl_type == 'string':
        if len(nodes.shape) == 1:
          feature = nodes.string_attrs[:, string_idx]
        elif len(nodes.shape) == 2:
          feature = nodes.string_attrs[:, :, string_idx]
        string_idx += 1
      else:
        raise ValueError('Unknown attr type %s' % attr_gl_type)
      feature = np.reshape(feature, [-1]).astype(attr_np_type)
      if attr_gl_type == 'string':
        feature = np.asarray(feature, order='C', dtype=object)
      features.append(feature)
    return features

  def _gen_pair(self, path, left_window_size, right_window_size):
    """Generate skip-gram pairs as positive pairs.

    Args:
      path: a list of ids start with root node's ids, each element is 1d numpy array
      with the same size.
    Returns:
      a pair of numpy array ids.

    Example:
    >>> path = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    >>> left_window_size = right_window_size = 1
    >>> src_id, dst_ids = self._gen_pair(path, left_window_size, right_window_size)
    >>> print print(src_ids, dst_ids)
    >>> (array([1, 2, 3, 4, 3, 4, 5, 6]), array([3, 4, 1, 2, 5, 6, 3, 4]))
    """
    path_len = len(path)
    pairs = [[], []]  # [src ids list, dst ids list]

    for center_idx in range(path_len):
      cursor = 0
      while center_idx - cursor > 0 and cursor < left_window_size:
        pairs[0].append(path[center_idx])
        pairs[1].append(path[center_idx - cursor - 1])
        cursor += 1

      cursor = 0
      while center_idx + cursor + 1 < path_len and cursor < right_window_size:
        pairs[0].append(path[center_idx])
        pairs[1].append(path[center_idx + cursor + 1])
        cursor += 1
    return np.concatenate(pairs[0]), np.concatenate(pairs[1])

  def _sample_generator(self):
    epoch_id = 0
    while self.num_epochs is None or epoch_id < self.num_epochs:
      # sample start nodes
      if self._mode == tf.estimator.ModeKeys.TRAIN:
        start_nodes = GraphInput.graph.V(GraphInput.node_type).batch(
            self._data_config.batch_size).alias('rand_walk_0')
      else:
        start_nodes = GraphInput.graph.V(GraphInput.node_type).batch(
            self._data_config.batch_size).alias('rand_walk_0')
      # sample paths
      for i in range(1, self._walk_len):
        out_alias = 'rand_walk_%d' % i
        start_nodes = start_nodes.outV(
            GraphInput.edge_type).sample(1).by('random').alias(out_alias)

      ds = gl.Dataset(start_nodes.values())

      while True:
        try:
          paths = ds.next()
          paths = [
              paths['rand_walk_%d' % i].ids.reshape([-1])
              for k in range(0, self._walk_len)
          ]
          # build positive pairs
          src_ids, dst_ids = self._gen_pair(paths, self._window_size,
                                            self._window_size)
          src_nodes = GraphInput.graph.get_nodes(GraphInput.node_type, src_ids)
          dst_nodes = GraphInput.graph.get_nodes(GraphInput.node_type, dst_ids)
          neg_nodes = self._neg_sampler.get(dst_ids)

          src_node_fea_arr = self._parse_nodes(src_nodes)
          dst_node_fea_arr = self._parse_nodes(dst_nodes)
          neg_node_fea_arr = self._parse_nodes(neg_nodes)

          yield tuple(src_node_fea_arr + dst_node_fea_arr + neg_node_fea_arr)
        except gl.OutOfRangeError:
          break
      if self._mode != tf.estimator.ModeKeys.TRAIN:
        break
      epoch_id += 1

  def _to_fea_dict(self, *features):
    fea_num = len(self._input_fields)
    assert fea_num * 3 == len(features)
    fea_dict_groups = {'src_fea': {}, 'positive_fea': {}, 'negative_fea': {}}
    for fid, fea in enumerate(features[:fea_num]):
      fea_dict_groups['src_fea'][self._input_fields[fid]] = fea
    for fid, fea in enumerate(features[fea_num:(fea_num * 2)]):
      fea_dict_groups['positive_fea'][self._input_fields[fid]] = fea
    for fid, fea in enumerate(features[(fea_num * 2):]):
      fea_dict_groups['negative_fea'][self._input_fields[fid]] = fea
    return fea_dict_groups

  def _group_preprocess(self, field_dict_groups):
    for g in field_dict_groups:
      field_dict_groups[g] = self._preprocess(field_dict_groups[g])
    return field_dict_groups

  def _get_labels(self, field_dict):
    return {
        'positive_fea': field_dict['positive_fea'],
        'negative_fea': field_dict['negative_fea']
    }

  def _get_features(self, field_dict_groups):
    return field_dict_groups['src_fea']

  def _build(self, mode, params):
    """Build graph dataset input for estimator.

    Args:
      mode: tf.estimator.ModeKeys.(TRAIN, EVAL, PREDICT)
      params: `dict` of hyper parameters, from Estimator

    Return:
      dataset: dataset for graph models.
    """
    self._mode = mode
    # get input type
    list_type = [self.get_tf_type(x) for x in self._input_field_types]
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)
    dataset = tf.data.Dataset.from_generator(
        self._sample_generator,
        output_types=list_type * 3,
        output_shapes=list_shapes * 3)

    # transform list to feature dict
    dataset = dataset.map(map_func=self._to_fea_dict)

    dataset = dataset.map(
        map_func=self._group_preprocess,
        num_parallel_calls=self._data_config.num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
