# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import division
from __future__ import print_function

import json
import logging
import math
import os
# import re
import threading

import numpy as np
import six
import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import ds_util
from easy_rec.python.utils.config_util import process_multi_file_input_path
from easy_rec.python.utils.tf_utils import get_tf_type

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


class BaseSampler(object):
  _instance_lock = threading.Lock()

  def __init__(self, fields, num_sample, num_eval_sample=None):
    self._g = None
    self._sampler = None
    self._num_sample = num_sample
    self._num_eval_sample = num_eval_sample if num_eval_sample is not None else num_sample
    self._build_field_types(fields)
    self._log_first_n = 5
    self._is_on_ds = ds_util.is_on_ds()

  def set_eval_num_sample(self):
    print('set_eval_num_sample: %d %d' %
          (self._num_sample, self._num_eval_sample))
    self._num_sample = self._num_eval_sample

  def _init_graph(self):
    if 'TF_CONFIG' in os.environ:
      tf_config = json.loads(os.environ['TF_CONFIG'])
      if 'ps' in tf_config['cluster']:
        # ps mode
        tf_config = json.loads(os.environ['TF_CONFIG'])
        task_count = len(tf_config['cluster']['worker']) + 2
        if self._is_on_ds:
          gl.set_tracker_mode(0)
          server_hosts = [
              host.split(':')[0] + ':888' + str(i)
              for i, host in enumerate(tf_config['cluster']['ps'])
          ]
          cluster = {
              'server': ','.join(server_hosts),
              'client_count': task_count
          }
        else:
          ps_count = len(tf_config['cluster']['ps'])
          cluster = {'server_count': ps_count, 'client_count': task_count}
        if tf_config['task']['type'] in ['chief', 'master']:
          self._g.init(cluster=cluster, job_name='client', task_index=0)
        elif tf_config['task']['type'] == 'worker':
          self._g.init(
              cluster=cluster,
              job_name='client',
              task_index=tf_config['task']['index'] + 2)
        # TODO(hongsheng.jhs): check cluster has evaluator or not?
        elif tf_config['task']['type'] == 'evaluator':
          self._g.init(
              cluster=cluster,
              job_name='client',
              task_index=tf_config['task']['index'] + 1)
          if self._num_eval_sample is not None and self._num_eval_sample > 0:
            self._num_sample = self._num_eval_sample
        elif tf_config['task']['type'] == 'ps':
          self._g.init(
              cluster=cluster,
              job_name='server',
              task_index=tf_config['task']['index'])
      else:
        # worker mode
        task_count = len(tf_config['cluster']['worker']) + 1
        if not self._is_on_ds:
          if tf_config['task']['type'] in ['chief', 'master']:
            self._g.init(task_index=0, task_count=task_count)
          elif tf_config['task']['type'] == 'worker':
            self._g.init(
                task_index=tf_config['task']['index'] + 1,
                task_count=task_count)
        else:
          gl.set_tracker_mode(0)
          if tf_config['cluster'].get('chief', ''):
            chief_host = tf_config['cluster']['chief'][0].split(
                ':')[0] + ':8880'
          else:
            chief_host = tf_config['cluster']['master'][0].split(
                ':')[0] + ':8880'
          worker_hosts = chief_host + [
              host.split(':')[0] + ':888' + str(i)
              for i, host in enumerate(tf_config['cluster']['worker'])
          ]

          if tf_config['task']['type'] in ['chief', 'master']:
            self._g.init(
                task_index=0,
                task_count=task_count,
                hosts=','.join(worker_hosts))
          elif tf_config['task']['type'] == 'worker':
            self._g.init(
                task_index=tf_config['task']['index'] + 1,
                task_count=task_count,
                hosts=worker_hosts)

        # TODO(hongsheng.jhs): check cluster has evaluator or not?
    else:
      # local mode
      self._g.init()

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
      self._attr_tf_types.append(get_tf_type(field.input_type))

  @classmethod
  def instance(cls, *args, **kwargs):
    with cls._instance_lock:
      if not hasattr(cls, '_instance'):
        cls._instance = cls(*args, **kwargs)
    return cls._instance

  def __del__(self):
    if self._g is not None:
      self._g.close()

  def _parse_nodes(self, nodes):
    if self._log_first_n > 0:
      logging.info('num_example=%d num_eval_example=%d node_num=%d' %
                   (self._num_sample, self._num_eval_sample, len(nodes.ids)))
      self._log_first_n -= 1
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
      if attr_gl_type == 'string':
        feature = feature.tolist()
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
      if attr_gl_type == 'string':
        feature = feature.tolist()
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
    ids = np.array(ids, dtype=np.int64)
    ids = np.pad(ids, (0, self._batch_size - len(ids)), 'edge')
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
      v.set_shape([self._num_sample])
      result_dict[k] = v
    return result_dict


class NegativeSamplerInMemory(BaseSampler):
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
    super(NegativeSamplerInMemory, self).__init__(fields, num_sample,
                                                  num_eval_sample)
    self._batch_size = batch_size

    self._item_ids = []
    self._cols = [[] for x in fields]

    if six.PY2 and isinstance(attr_delimiter, type(u'')):
      attr_delimiter = attr_delimiter.encode('utf-8')
    if data_path.startswith('odps://'):
      self._load_table(data_path, attr_delimiter)
    else:
      self._load_data(data_path, attr_delimiter)

    print('NegativeSamplerInMemory: total_row_num = %d' % len(self._cols[0]))
    for col_id in range(len(self._attr_np_types)):
      np_type = self._attr_np_types[col_id]
      print('\tcol_id[%d], dtype=%s' % (col_id, self._attr_gl_types[col_id]))
      if np_type != np.str:
        self._cols[col_id] = np.array(self._cols[col_id], dtype=np_type)
      else:
        self._cols[col_id] = np.asarray(
            self._cols[col_id], order='C', dtype=object)

  def _load_table(self, data_path, attr_delimiter):
    import common_io
    reader = common_io.table.TableReader(data_path)
    schema = reader.get_schema()
    item_id_col = 0
    fea_id_col = 2
    for tid in range(len(schema)):
      if schema[tid][0].startswith('feature'):
        fea_id_col = tid
        break
    for tid in range(len(schema)):
      if schema[tid][0].startswith('id'):
        item_id_col = tid
        break
    print('NegativeSamplerInMemory: feature_id_col = %d, item_id_col = %d' %
          (fea_id_col, item_id_col))
    while True:
      try:
        row_arr = reader.read(num_records=1024, allow_smaller_final_batch=True)
        for row in row_arr:
          # item_id, weight, feature
          self._item_ids.append(int(row[item_id_col]))
          col_vals = row[fea_id_col].split(attr_delimiter)
          assert len(col_vals) == len(
              self._cols), 'invalid row[%d %d]: %s %s' % (len(
                  col_vals), len(self._cols), row[item_id_col], row[fea_id_col])
          for col_id in range(len(col_vals)):
            self._cols[col_id].append(col_vals[col_id])
      except common_io.exception.OutOfRangeException:
        reader.close()
        break

  def _load_data(self, data_path, attr_delimiter):
    item_id_col = 0
    fea_id_col = 2
    print('NegativeSamplerInMemory: load sample feature from %s' % data_path)
    with tf.gfile.GFile(data_path, 'r') as fin:
      for line_id, line_str in enumerate(fin):
        line_str = line_str.strip()
        cols = line_str.split('\t')
        if line_id == 0:
          schema = [x.split(':') for x in cols]
          for tid in range(len(schema)):
            if schema[tid][0].startswith('id'):
              item_id_col = tid
            if schema[tid][0].startswith('feature'):
              fea_id_col = tid
          print('feature_id_col = %d, item_id_col = %d' %
                (fea_id_col, item_id_col))
        else:
          self._item_ids.append(int(cols[item_id_col]))
          fea_vals = cols[fea_id_col].split(attr_delimiter)
          assert len(fea_vals) == len(
              self._cols), 'invalid row[%d][%d %d]:%s %s' % (
                  line_id, len(fea_vals), len(
                      self._cols), cols[item_id_col], cols[fea_id_col])
          for col_id in range(len(fea_vals)):
            self._cols[col_id].append(fea_vals[col_id])

  def _get_impl(self, ids):
    features = []
    if type(ids[0]) != int:
      ids = [int(x) for x in ids]
    assert self._num_sample > 0, 'invalid num_sample: %d' % self._num_sample

    indices = np.random.choice(
        len(self._item_ids),
        size=self._num_sample + self._batch_size,
        replace=False)

    sel_ids = []
    for tid in indices:
      rid = self._item_ids[tid]
      if rid not in ids:
        sel_ids.append(tid)
        if len(sel_ids) >= self._num_sample and self._num_sample > 0:
          break

    features = []
    for col_id in range(len(self._cols)):
      tmp_col = self._cols[col_id]
      np_type = self._attr_np_types[col_id]
      if np_type != np.str:
        sel_feas = tmp_col[sel_ids]
        features.append(sel_feas)
      else:
        features.append(
            np.asarray([tmp_col[x] for x in sel_ids], order='C', dtype=object))
    return features

  def get(self, ids):
    """Sampling method.

    Args:
      ids: item id tensor.

    Returns:
      Negative sampled feature dict.
    """
    all_attr_types = list(self._attr_tf_types)
    if self._num_sample <= 0:
      all_attr_types.append(tf.float32)
    sampled_values = tf.py_func(self._get_impl, [ids], all_attr_types)
    result_dict = {}
    for k, v in zip(self._attr_names, sampled_values):
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
    src_ids = np.pad(src_ids, (0, self._batch_size - len(src_ids)), 'edge')
    dst_ids = np.array(dst_ids, dtype=np.int64)
    dst_ids = np.pad(dst_ids, (0, self._batch_size - len(dst_ids)), 'edge')
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
    dst_ids = np.pad(dst_ids, (0, self._batch_size - len(dst_ids)), 'edge')
    nodes = self._neg_sampler.get(dst_ids)
    neg_features = self._parse_nodes(nodes)
    sparse_nodes = self._hard_neg_sampler.get(src_ids).layer_nodes(1)
    hard_neg_features, hard_neg_indices = self._parse_sparse_nodes(sparse_nodes)

    results = []
    for i, v in enumerate(hard_neg_features):
      if type(v) == list:
        results.append(np.asarray(neg_features[i] + v, order='C', dtype=object))
      else:
        results.append(np.concatenate([neg_features[i], v], axis=0))
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
              decoder=gl.Decoder(weighted=True)) \
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
    src_ids_padded = np.pad(src_ids, (0, self._batch_size - len(src_ids)),
                            'edge')
    dst_ids = np.array(dst_ids, dtype=np.int64)
    dst_ids = np.pad(dst_ids, (0, self._batch_size - len(dst_ids)), 'edge')
    nodes = self._neg_sampler.get(src_ids_padded, dst_ids)
    neg_features = self._parse_nodes(nodes)
    sparse_nodes = self._hard_neg_sampler.get(src_ids).layer_nodes(1)
    hard_neg_features, hard_neg_indices = self._parse_sparse_nodes(sparse_nodes)

    results = []
    for i, v in enumerate(hard_neg_features):
      if type(v) == list:
        results.append(np.asarray(neg_features[i] + v, order='C', dtype=object))
      else:
        results.append(np.concatenate([neg_features[i], v], axis=0))
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
  print('sampler_type = %s' % sampler_type)
  sampler_config = getattr(data_config, sampler_type)

  if ds_util.is_on_ds():
    gl.set_field_delimiter(sampler_config.field_delimiter)

  if sampler_type == 'negative_sampler':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]

    input_path = process_multi_file_input_path(sampler_config.input_path)
    return NegativeSampler.instance(
        data_path=input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'negative_sampler_in_memory':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]

    input_path = process_multi_file_input_path(sampler_config.input_path)
    return NegativeSamplerInMemory.instance(
        data_path=input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'negative_sampler_v2':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]

    user_input_path = process_multi_file_input_path(
        sampler_config.user_input_path)
    item_input_path = process_multi_file_input_path(
        sampler_config.item_input_path)
    pos_edge_input_path = process_multi_file_input_path(
        sampler_config.pos_edge_input_path)
    return NegativeSamplerV2.instance(
        user_data_path=user_input_path,
        item_data_path=item_input_path,
        edge_data_path=pos_edge_input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'hard_negative_sampler':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]

    user_input_path = process_multi_file_input_path(
        sampler_config.user_input_path)
    item_input_path = process_multi_file_input_path(
        sampler_config.item_input_path)
    hard_neg_edge_input_path = process_multi_file_input_path(
        sampler_config.hard_neg_edge_input_path)
    return HardNegativeSampler.instance(
        user_data_path=user_input_path,
        item_data_path=item_input_path,
        hard_neg_edge_data_path=hard_neg_edge_input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        num_hard_sample=sampler_config.num_hard_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  elif sampler_type == 'hard_negative_sampler_v2':
    input_fields = {f.input_name: f for f in data_config.input_fields}
    attr_fields = [input_fields[name] for name in sampler_config.attr_fields]

    user_input_path = process_multi_file_input_path(
        sampler_config.user_input_path)
    item_input_path = process_multi_file_input_path(
        sampler_config.item_input_path)
    pos_edge_input_path = process_multi_file_input_path(
        sampler_config.pos_edge_input_path)
    hard_neg_edge_input_path = process_multi_file_input_path(
        sampler_config.hard_neg_edge_input_path)
    return HardNegativeSamplerV2.instance(
        user_data_path=user_input_path,
        item_data_path=item_input_path,
        edge_data_path=pos_edge_input_path,
        hard_neg_edge_data_path=hard_neg_edge_input_path,
        fields=attr_fields,
        num_sample=sampler_config.num_sample,
        num_hard_sample=sampler_config.num_hard_sample,
        batch_size=data_config.batch_size,
        attr_delimiter=sampler_config.attr_delimiter,
        num_eval_sample=sampler_config.num_eval_sample)
  else:
    raise ValueError('Unknown sampler %s' % sampler_type)
