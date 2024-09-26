# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Common functions used for odps input."""
import json
import os

import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def get_ps_num_from_tf_config():
  tf_config = os.environ.get('TF_CONFIG')
  if tf_config:
    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster', {})
    ps_hosts = cluster.get('ps', [])
    return len(ps_hosts)
  return 0


def get_tf_type(field_type):
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


def get_col_type(tf_type):
  type_map = {
      tf.int32: 'BIGINT',
      tf.int64: 'BIGINT',
      tf.string: 'STRING',
      tf.float32: 'FLOAT',
      tf.double: 'DOUBLE',
      tf.bool: 'BOOLEAN'
  }
  assert tf_type in type_map, 'invalid type: %s' % tf_type
  return type_map[tf_type]


def add_elements_to_collection(elements, name):
  collection = tf.get_collection_ref(name)
  collection_set = set(collection)
  for element in elements:
    if element not in collection_set:
      collection.append(element)
