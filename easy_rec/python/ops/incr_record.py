# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import logging
import easy_rec
import tensorflow as tf

op_path = os.path.join(easy_rec.ops_dir, "incr_record.so")
try:
  op = tf.load_op_library(op_path)
  get_sparse_indices = op.get_sparse_indices
  set_sparse_indices = op.set_sparse_indices
  del op
except ImportError as ex:
  get_sparse_indices = None
  set_sparse_indices = None
  logging.warning('failed to import gen_io_ops.collect_sparse_indices: %s' % str(ex))
