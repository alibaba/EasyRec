# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import tensorflow as tf
from tensorflow.python.ops import string_ops

import easy_rec
from easy_rec.python.utils import constant

try:
  str_avx_op_path = os.path.join(easy_rec.ops_dir, 'libstr_avx_op.so')
  str_avx_op = tf.load_op_library(str_avx_op_path)
  logging.info('load avx string_split op from %s succeed' % str_avx_op_path)
except Exception as ex:
  logging.warning('load avx string_split op failed: %s' % str(ex))
  str_avx_op = None


def str_split_by_chr(input_str, sep, skip_empty):
  if constant.has_avx_str_split() and str_avx_op is not None:
    assert len(sep) == 1, \
        'invalid data_config.separator(%s) len(%d) != 1' % (
        sep, len(sep))
    return str_avx_op.avx512_string_split(input_str, sep, skip_empty=skip_empty)
  else:
    return string_ops.string_split(input_str, sep, skip_empty=skip_empty)
