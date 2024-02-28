# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

SAMPLE_WEIGHT = 'SAMPLE_WEIGHT'

DENSE_UPDATE_VARIABLES = 'DENSE_UPDATE_VARIABLES'

SPARSE_UPDATE_VARIABLES = 'SPARSE_UPDATE_VARIABLES'
ENABLE_AVX_STR_SPLIT = 'ENABLE_AVX_STR_SPLIT'

# Environment variables to control whether to sort
# feature columns by name, by default sort is not
# enabled. The flag is set for backward compatibility.
SORT_COL_BY_NAME = 'SORT_COL_BY_NAME'

# arithmetic_optimization causes significant slow training
# of a test case:
#   train_eval_test.TrainEvalTest.test_train_parquet
NO_ARITHMETRIC_OPTI = 'NO_ARITHMETRIC_OPTI'

# shard embedding var_name collection
EmbeddingParallel = 'EmbeddingParallel'

# environ variable to force embedding placement on cpu
EmbeddingOnCPU = 'place_embedding_on_cpu'


def enable_avx_str_split():
  os.environ[ENABLE_AVX_STR_SPLIT] = '1'


def has_avx_str_split():
  return ENABLE_AVX_STR_SPLIT in os.environ and os.environ[
      ENABLE_AVX_STR_SPLIT] == '1'


def disable_avx_str_split():
  del os.environ[ENABLE_AVX_STR_SPLIT]
