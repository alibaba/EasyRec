# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.protos.train_pb2 import DistributionStrategy
from easy_rec.python.protos.train_pb2 import TrainConfig


def build(train_config):
  assert isinstance(train_config, TrainConfig)

  distribution = None
  # single worker multi-gpu strategy
  # currently only works using pai-tf
  if train_config.train_distribute == DistributionStrategy.MirroredStrategy:
    if tf.__version__ <= '1.15':
      distribution = tf.contrib.distribute.MirroredStrategy()
    else:
      distribution = tf.distribute.MirroredStrategy()
  # multi worker multi-gpu strategy
  # works under tf1.15 and tf2.x
  elif train_config.train_distribute == DistributionStrategy.MultiWorkerMirroredStrategy:
    distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  # only works using pai-tf
  elif train_config.train_distribute == DistributionStrategy.ExascaleStrategy:
    import pai
    distribution = pai.distribute.ExascaleStrategy(
        max_splits=10,
        issorted=True,
        optimize_clip_by_global_norm=False,
        enable_sparse_allreduce=False,
        enable_hierarchical_allreduce=True)
  # the older version of MultiWorkerMirroredStrategy
  # works under tf1.12 to tf1.15
  elif train_config.train_distribute == DistributionStrategy.CollectiveAllReduceStrategy:
    distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
        num_gpus_per_worker=train_config.num_gpus_per_worker)
  # works under tf1.15 and tf2.x
  elif train_config.train_distribute == DistributionStrategy.PSStrategy:
    if tf.__version__ <= '1.15':
      distribution = tf.contrib.distribute.ParameterServerStrategy()
    else:
      distribution = tf.distribute.experimental.ParameterServerStrategy()
  return distribution
