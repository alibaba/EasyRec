# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

SAMPLE_WEIGHT = 'SAMPLE_WEIGHT'

DENSE_UPDATE_VARIABLES = 'DENSE_UPDATE_VARIABLES'

SPARSE_UPDATE_VARIABLES = 'SPARSE_UPDATE_VARIABLES'
ENABLE_AVX_STR_SPLIT = 'ENABLE_AVX_STR_SPLIT'


def enable_avx_str_split():
  os.environ[ENABLE_AVX_STR_SPLIT] = '1'


def has_avx_str_split():
  return ENABLE_AVX_STR_SPLIT in os.environ and os.environ[
      ENABLE_AVX_STR_SPLIT] == '1'


def disable_avx_str_split():
  del os.environ[ENABLE_AVX_STR_SPLIT]
