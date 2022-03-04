# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import threading

import numpy as np
import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import graph_utils

try:
  import graphlearn as gl
except Exception:
  logging.info(
      'GraphLearn is not installed. You can install it by "pip install https://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/graphlearn-0.7-cp27-cp27mu-linux_x86_64.whl"'  # noqa: E501
  )

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def _get_gl_type(field_type):
  type_map = {
      DatasetConfig.INT32: 'int',
      DatasetConfig.INT64: 'int',
      DatasetConfig.STRING: 'string',
      DatasetConfig.BOOL: 'int',
      DatasetConfig.FLOAT: 'float',
      DatasetConfig.DOUBLE: 'float'
  }
  assert field_type in type_map, 'invalid type: %s' % field_type
  return type_map[field_type]


def _get_np_type(field_type):
  type_map = {
      DatasetConfig.INT32: np.int32,
      DatasetConfig.INT64: np.int64,
      DatasetConfig.STRING: np.str,
      DatasetConfig.BOOL: np.bool,
      DatasetConfig.FLOAT: np.float32,
      DatasetConfig.DOUBLE: np.double
  }
  assert field_type in type_map, 'invalid type: %s' % field_type
  return type_map[field_type]


def _get_tf_type(field_type):
  type_map = {
      DatasetConfig.INT32: tf.int32,
      DatasetConfig.INT64: tf.int64,
      DatasetConfig.STRING: tf.string,
      DatasetConfig.BOOL: tf.bool,
      DatasetConfig.FLOAT: tf.float32,
      DatasetConfig.DOUBLE: tf.double
  }
  assert field_type in type_map, 'invalid type: %s' % field_type
  return type_map[field_type]


class BaseSampler(object):
  _instance_lock = threading.Lock()

  def __init__(self, fields, num_sample, num_eval_sample=None):
    self._g = None
    self._sampler = None
    # TODO(hongsheng.jhs): check eval mode or not?
    self._num_sample = num_sample
    self._num_eval_sample = num_eval_sample if num_eval_sample else num_sample
    self._build_field_types(fields)

  def _init_graph(self):
    graph_utils.graph_init(self._g, os.environ.get('TF_CONFIG', None))

  def _build_field_types(self, fields):
    self._attr_names = []
    self._attr_types = []
    self._attr_gl_types = []
    self._attr_np_types = []
    self._attr_tf_types = []
    for i, field in enumerate(fields):
      self._attr_names.append(field.input_name)
      self._attr_types.append(field.input_type)
      self._attr_gl_types.append(_get_gl_type(field.input_type))
      self._attr_np_types.append(_get_np_type(field.input_type))
      self._attr_tf_types.append(_get_tf_type(field.input_type))

  @classmethod
  def instance(cls, *args, **kwargs):
    with cls._instance_lock:
      if not hasattr(cls, '_instance'):
        cls._instance = cls(*args, **kwargs)
    return cls._instance

  def __del__(self):
    self._g.close()

  def _parse_nodes(self, nodes):
    features = []
    int_idx = 0
    float_idx = 0
    string_idx = 0
    for attr_gl_type, attr_np_type in zip(self._attr_gl_types,
                                          self._attr_np_types):
      if attr_gl_type == 'int':
        feature = nodes.int_attrs[:, :, int_idx]
        int_idx += 1
      elif attr_gl_type == 'float':
        feature = nodes.float_attrs[:, :, float_idx]
        float_idx += 1
      elif attr_gl_type == 'string':
        feature = nodes.string_attrs[:, :, string_idx]
        string_idx += 1
      else:
        raise ValueError('Unknown attr type %s' % attr_gl_type)
      feature = np.reshape(feature,
                           [-1])[:self._num_sample].astype(attr_np_type)
      features.append(feature)
    return features

  def _parse_sparse_nodes(self, nodes):
    features = []
    int_idx = 0
    float_idx = 0
    string_idx = 0
    for attr_gl_type, attr_np_type in zip(self._attr_gl_types,
                                          self._attr_np_types):
      if attr_gl_type == 'int':
        feature = nodes.int_attrs[:, int_idx]
        int_idx += 1
      elif attr_gl_type == 'float':
        feature = nodes.float_attrs[:, float_idx]
        float_idx += 1
      elif attr_gl_type == 'string':
        feature = nodes.string_attrs[:, string_idx]
        string_idx += 1
      else:
        raise ValueError('Unknown attr type %s' % attr_gl_type)
      feature = feature.astype(attr_np_type)
      features.append(feature)
    return features, nodes.indices


class NegativeSampler(BaseSampler):
  """Negative Sampler.

  Weighted random sampling items not in batch.

  Args:
    data_path: item feature data path. id:int64 | weight:float | attrs:string.
    fields: item input fields.
    num_sample: number of negative samples.
    batch_size: mini-batch size.
    attr_delimiter: delimiter of feature string.
    num_eval_sample: number of negative samples for evaluator.
  """

  def __init__(self,
               data_path,
               fields,
               num_sample,
               batch_size,
               attr_delimiter=':',
               num_eval_sample=None):
    super(NegativeSampler, self).__init__(fields, num_sample, num_eval_sample)
    self._batch_size = batch_size
    self._g = gl.Graph().node(
        tf.compat.as_str(data_path),
        node_type='item',
        decoder=gl.Decoder(
            attr_types=self._attr_gl_types,
            weighted=True,
            attr_delimiter=attr_delimiter))
    self._init_graph()

    expand_factor = int(math.ceil(self._num_sample / batch_size))
    self._sampler = self._g.negative_sampler(
        'item', expand_factor, strategy='node_weight')

  def _get_impl(self, ids):
    # assert len(ids) == self._batch_size
    # tf.logging.info("ids: %s", len(ids))
    ids = np.array(ids, dtype=np.int64)
    nodes = self._sampler.get(ids)
    features = self._parse_nodes(nodes)
    return features

  def get(self, ids):
    """Sampling method.

    Args:
      ids: item id tensor.

    Returns:
      Negative sampled feature dict.
    """
    sampled_values = tf.py_func(self._get_impl, [ids], self._attr_tf_types)
    result_dict = {}
    for k, t, v in zip(self._attr_names, self._attr_tf_types, sampled_values):
      if t == tf.string:
        # string convert from np array to tensor will be padded with \000, we need remove it
        v = tf.regex_replace(v, '\000', '')
      v.set_shape([self._num_sample])
      result_dict[k] = v
    return result_dict


class NegativeSamplerV2(BaseSampler):
  """Negative Sampler V2.

  Weighted random sampling items which do not have positive edge with the user.

  Args:
    user_data_path: user node data path. id:int64 | weight:float.
    item_data_path: item feature data path. id:int64 | weight:float | attrs:string.
    edge_data_path: positive edge data path. userid:int64 | itemid:int64 | weight:float
    fields: item input fields.
    num_sample: number of negative samples.
    batch_size: mini-batch size.
    attr_delimiter: delimiter of feature string.
    num_eval_sample: number of negative samples for evaluator.
  """

  def __init__(self,
               user_data_path,
               item_data_path,
               edge_data_path,
               fields,
               num_sample,
               batch_size,
               attr_delimiter=':',
               num_eval_sample=None):
    super(NegativeSamplerV2, self).__init__(fields, num_sample, num_eval_sample)
    self._batch_size = batch_size
    self._g = gl.Graph() \
        .node(tf.compat.as_str(user_data_path),
              node_type='user',
              decoder=gl.Decoder(weighted=True)) \
        .node(tf.compat.as_str(item_data_path),
              node_type='item',
              decoder=gl.Decoder(
                  attr_types=self._attr_gl_types,
                  weighted=True,
                  attr_delimiter=attr_delimiter)) \
        .edge(tf.compat.as_str(edge_data_path),
              edge_type=('user', 'item', 'edge'),
              decoder=gl.Decoder(weighted=True))
    self._init_graph()

    expand_factor = int(math.ceil(self._num_sample / batch_size))
    self._sampler = self._g.negative_sampler(
        'edge', expand_factor, strategy='random', conditional=True)

  def _get_impl(self, src_ids, dst_ids):
    src_ids = np.array(src_ids, dtype=np.int64)
    dst_ids = np.array(dst_ids, dtype=np.int64)
    nodes = self._sampler.get(src_ids, dst_ids)
    features = self._parse_nodes(nodes)
    return features

  def get(self, src_ids, dst_ids):
    """Sampling method.

    Args:
      src_ids: user id tensor.
      dst_ids: item id tensor.

    Returns:
      Negative sampled feature dict.
    """
    sampled_values = tf.py_func(self._get_impl, [src_ids, dst_ids],
                                self._attr_tf_types)
    result_dict = {}
    for k, t, v in zip(self._attr_names, self._attr_tf_types, sampled_values):
      if t == tf.string:
        # string convert from np array to tensor will be padded with \000, we need remove it
        v = tf.regex_replace(v, '\000', '')
      v.set_shape([self._num_sample])
      result_dict[k] = v
    return result_dict


class HardNegativeSampler(BaseSampler):
  """HardNegativeSampler.

  Weighted random sampling items not in batch as negative samples, and sampling
  destination nodes in hard_neg_edge as hard negative samples

  Args:
    user_data_path: user node data path. id:int64 | weight:float.
    item_data_path: item feature data path. id:int64 | weight:float | attrs:string.
    hard_neg_edge_data_path: hard negative edge data path. userid:int64 | itemid:int64 | weight:float
    fields: item input fields.
    num_sample: number of negative samples.
    num_hard_sample: maximum number of hard negative samples.
    batch_size: mini-batch size.
    attr_delimiter: delimiter of feature string.
    num_eval_sample: number of negative samples for evaluator.
  """

  def __init__(self,
               user_data_path,
               item_data_path,
               hard_neg_edge_data_path,
               fields,
               num_sample,
               num_hard_sample,
               batch_size,
               attr_delimiter=':',
               num_eval_sample=None):
    super(HardNegativeSampler, self).__init__(fields, num_sample,
                                              num_eval_sample)
    self._batch_size = batch_size
    self._g = gl.Graph() \
        .node(tf.compat.as_str(user_data_path),
              node_type='user',
              decoder=gl.Decoder(weighted=True)) \
        .node(tf.compat.as_str(item_data_path),
              node_type='item',
              decoder=gl.Decoder(
                  attr_types=self._attr_gl_types,
                  weighted=True,
                  attr_delimiter=attr_delimiter)) \
        .edge(tf.compat.as_str(hard_neg_edge_data_path),
              edge_type=('user', 'item', 'hard_neg_edge'),
              decoder=gl.Decoder(weighted=True))
    self._init_graph()

    expand_factor = int(math.ceil(self._num_sample / batch_size))
    self._neg_sampler = self._g.negative_sampler(
        'item', expand_factor, strategy='node_weight')
    self._hard_neg_sampler = self._g.neighbor_sampler(['hard_neg_edge'],
                                                      num_hard_sample,
                                                      strategy='full')

  def _get_impl(self, src_ids, dst_ids):
    src_ids = np.array(src_ids, dtype=np.int64)
    dst_ids = np.array(dst_ids, dtype=np.int64)
    nodes = self._neg_sampler.get(dst_ids)
    neg_features = self._parse_nodes(nodes)
    sparse_nodes = self._hard_neg_sampler.get(src_ids).layer_nodes(1)
    hard_neg_features, hard_neg_indices = self._parse_sparse_nodes(sparse_nodes)

    results = []
    for i, v in enumerate(hard_neg_features):
      results.append(np.concatenate([neg_features[i], v], axis=-1))
    results.append(hard_neg_indices)
    return results

  def get(self, src_ids, dst_ids):
    """Sampling method.

    Args:
      src_ids: user id tensor.
      dst_ids: item id tensor.

    Returns:
      Sampled feature dict. The first batch_size is negative samples, remainder is hard negative samples
    """
    output_types = self._attr_tf_types + [tf.int64]
    output_values = tf.py_func(self._get_impl, [src_ids, dst_ids], output_types)
    result_dict = {}
    for k, t, v in zip(self._attr_names, self._attr_tf_types,
                       output_values[:-1]):
      if t == tf.string:
        # string convert from np array to tensor will be padded with \000, we need remove it
        v = tf.regex_replace(v, '\000', '')
      v.set_shape([None])
      result_dict[k] = v

    hard_neg_indices = output_values[-1]
    hard_neg_indices.set_shape([None, 2])
    result_dict['hard_neg_indices'] = hard_neg_indices
    return result_dict


class HardNegativeSamplerV2(BaseSampler):
  """HardNegativeSampler.

  Weighted random sampling items which  do not have positive edge with the user., and sampling
  destination nodes in hard_neg_edge as hard negative samples

  Args:
    user_data_path: user node data path. id:int64 | weight:float.
    item_data_path: item feature data path. id:int64 | weight:float | attrs:string.
    edge_data_path: positive  edge data path. userid:int64 | itemid:int64 | weight:float
    hard_neg_edge_data_path: hard negative edge data path. userid:int64 | itemid:int64 | weight:float
    fields: item input fields.
    num_sample: number of negative samples.
    num_hard_sample: maximum number of hard negative samples.
    batch_size: mini-batch size.
    attr_delimiter: delimiter of feature string.
    num_eval_sample: number of negative samples for evaluator.
  """

  def __init__(self,
               user_data_path,
               item_data_path,
               edge_data_path,
               hard_neg_edge_data_path,
               fields,
               num_sample,
               num_hard_sample,
               batch_size,
               attr_delimiter=':',
               num_eval_sample=None):
    super(HardNegativeSamplerV2, self).__init__(fields, num_sample,
                                                num_eval_sample)
    self._batch_size = batch_size
    self._g = gl.Graph() \
        .node(tf.compat.as_str(user_data_path),
              node_type='user',
              decoder=gl.Decoder(weighted=True)) \
        .node(tf.compat.as_str(item_data_path),
              node_type='item',
              decoder=gl.Decoder(
                  attr_types=self._attr_gl_types,
                  weighted=True,
                  attr_delimiter=attr_delimiter)) \
        .edge(tf.compat.as_str(edge_data_path),
              edge_type=('user', 'item', 'edge'),
              decoder=gl.Decoder(weighted=True))  \
        .edge(tf.compat.as_str(hard_neg_edge_data_path),
              edge_type=('user', 'item', 'hard_neg_edge'),
              decoder=gl.Decoder(weighted=True))
    self._init_graph()

    expand_factor = int(math.ceil(self._num_sample / batch_size))
    self._neg_sampler = self._g.negative_sampler(
        'edge', expand_factor, strategy='random', conditional=True)
    self._hard_neg_sampler = self._g.neighbor_sampler(['hard_neg_edge'],
                                                      num_hard_sample,
                                                      strategy='full')

  def _get_impl(self, src_ids, dst_ids):
    src_ids = np.array(src_ids, dtype=np.int64)
    dst_ids = np.array(dst_ids, dtype=np.int64)
    nodes = self._neg_sampler.get(src_ids, dst_ids)
    neg_features = self._parse_nodes(nodes)
    sparse_nodes = self._hard_neg_sampler.get(src_ids).layer_nodes(1)
    hard_neg_features, hard_neg_indices = self._parse_sparse_nodes(sparse_nodes)

    results = []
    for i, v in enumerate(hard_neg_features):
      results.append(np.concatenate([neg_features[i], v], axis=-1))
    results.append(hard_neg_indices)
    return results

  def get(self, src_ids, dst_ids):
    """Sampling method.

    Args:
      src_ids: user id tensor.
      dst_ids: item id tensor.

    Returns:
      Sampled feature dict. The first batch_size is negative samples, remainder is hard negative samples
    """
    output_types = self._attr_tf_types + [tf.int64]
    output_values = tf.py_func(self._get_impl, [src_ids, dst_ids], output_types)
    result_dict = {}
    for k, t, v in zip(self._attr_names, self._attr_tf_types,
                       output_values[:-1]):
      if t == tf.string:
        # string convert from np array to tensor will be padded with \000, we need remove it
        v = tf.regex_replace(v, '\000', '')
      v.set_shape([None])
      result_dict[k] = v

    hard_neg_indices = output_values[-1]
    hard_neg_indices.set_shape([None, 2])
    result_dict['hard_neg_indices'] = hard_neg_indices
    return result_dict


def build(data_config):
  if not data_config.HasField('sampler'):
    return None
  sampler_type = data_config.WhichOneof('sampler')
  sampler_config = getattr(data_config, sampler_type)
  if sampler_type == 'negative_sampler':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]
    return NegativeSampler.instance(
        data_path=sampler_config.input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'negative_sampler_v2':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]
    return NegativeSamplerV2.instance(
        user_data_path=sampler_config.user_input_path,
        item_data_path=sampler_config.item_input_path,
        edge_data_path=sampler_config.pos_edge_input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'hard_negative_sampler':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]
    return HardNegativeSampler.instance(
        user_data_path=sampler_config.user_input_path,
        item_data_path=sampler_config.item_input_path,
        hard_neg_edge_data_path=sampler_config.hard_neg_edge_input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        num_hard_sample=sampler_config.num_hard_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'hard_negative_sampler_v2':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]
    return HardNegativeSamplerV2.instance(
        user_data_path=sampler_config.user_input_path,
        item_data_path=sampler_config.item_input_path,
        edge_data_path=sampler_config.pos_edge_input_path,
        hard_neg_edge_data_path=sampler_config.hard_neg_edge_input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        num_hard_sample=sampler_config.num_hard_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  else:
    raise ValueError('Unknown sampler %s' % sampler_type)
