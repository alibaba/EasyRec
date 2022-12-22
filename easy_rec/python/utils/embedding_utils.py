# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.variables import global_variables
from tensorflow.python.platform import gfile

from easy_rec.python.utils import constant
from easy_rec.python.utils import proto_util

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def get_norm_name_to_ids():
  """Get normalize embedding name(including kv variables) to ids.

  Return:
    normalized names to ids mapping.
  """
  norm_name_to_ids = {}
  for x in global_variables():
    if 'EmbeddingVariable' in str(type(x)):
      norm_name, _ = proto_util.get_norm_embed_name(x.name)
    elif '/embedding_weights:' in x.name or '/embedding_weights/part_' in x.name:
      norm_name, _ = proto_util.get_norm_embed_name(x.name)
      norm_name_to_ids[norm_name] = 1
  for tid, t in enumerate(norm_name_to_ids.keys()):
    norm_name_to_ids[t] = str(tid)
  return norm_name_to_ids


def save_norm_name_to_ids(save_path, norm_name_to_ids):
  """Save normalize name to ids mapping.

  Args:
    save_path: save path.
    norm_name_to_ids: dict, map normalized name to ids.
  """
  with gfile.GFile(save_path, 'w') as fout:
    for k in norm_name_to_ids:
      fout.write('%s\t%s\n' % (k, norm_name_to_ids[k]))


def load_norm_name_to_ids(save_path):
  """Load normalize name to ids mapping from file.

  Args:
    save_path: file path.

  Return:
    dict, map normalized name to ids.
  """
  norm_name_to_ids = {}
  with gfile.GFile(save_path, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      k, v = line_str.split('\t')
      norm_name_to_ids[k] = int(v)
  return norm_name_to_ids


def get_sparse_name_to_ids(norm_name_to_ids):
  """Get embedding variable(including kv variables) name to ids mapping.

  Return:
     dict, normalized variable names to ids mappping.
  """
  name_to_ids = {}
  for x in ops.get_collection(constant.SPARSE_UPDATE_VARIABLES):
    norm_name, _ = proto_util.get_norm_embed_name(x[0].name)
    name_to_ids[x[0].name] = norm_name_to_ids[norm_name]
  return name_to_ids


def get_dense_name_to_ids():
  """Get dense variable(embedding excluded) name to ids mapping.

  Return:
    dict, dense variable names to ids mapping.
  """
  dense_train_vars = ops.get_collection(constant.DENSE_UPDATE_VARIABLES)
  norm_name_to_ids = {}
  for tid, x in enumerate(dense_train_vars):
    norm_name_to_ids[x.op.name] = tid
  return norm_name_to_ids
