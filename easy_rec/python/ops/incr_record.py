# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import tensorflow as tf

import easy_rec

try:
  op_path = os.path.join(easy_rec.ops_dir, 'incr_record.so')
  op = tf.load_op_library(op_path)
  get_sparse_indices = op.get_sparse_indices
  set_sparse_indices = op.set_sparse_indices
  if 'kv_resource_incr_gather' in dir(op):
    kv_resource_incr_gather = getattr(op, 'kv_resource_incr_gather')
  else:
    kv_resource_incr_gather = None
except ImportError as ex:
  get_sparse_indices = None
  set_sparse_indices = None
  kv_resource_incr_gather = None
  logging.warning('failed to import gen_io_ops.collect_sparse_indices: %s' %
                  str(ex))
except Exception as ex:
  get_sparse_indices = None
  set_sparse_indices = None
  kv_resource_incr_gather = None
  logging.warning('failed to import gen_io_ops.collect_sparse_indices: %s' %
                  str(ex))
