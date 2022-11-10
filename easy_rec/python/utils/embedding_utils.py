# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.python.framework import ops

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
  for x in ops.get_collection(constant.SPARSE_UPDATE_VARIABLES):
    norm_name, part_id = proto_util.get_norm_embed_name(x[0].name)
    norm_name_to_ids[norm_name] = 1

  for tid, t in enumerate(norm_name_to_ids.keys()):
    norm_name_to_ids[t] = str(tid)
  return norm_name_to_ids


def get_sparse_name_to_ids():
  """Get embedding variable(including kv variables) name to ids mapping.

  Return:
     variable names to ids mappping.
  """
  norm_name_to_ids = get_norm_name_to_ids()
  name_to_ids = {}
  for x in ops.get_collection(constant.SPARSE_UPDATE_VARIABLES):
    norm_name, _ = proto_util.get_norm_embed_name(x[0].name)
    name_to_ids[x[0].name] = norm_name_to_ids[norm_name]
  return name_to_ids


def get_dense_name_to_ids():
  dense_train_vars = ops.get_collection(constant.DENSE_UPDATE_VARIABLES)
  norm_name_to_ids = {}
  for tid, x in enumerate(dense_train_vars):
    norm_name_to_ids[x.op.name] = tid
  return norm_name_to_ids
