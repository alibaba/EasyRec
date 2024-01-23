# -*- encoding: utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook

from easy_rec.python.utils import constant

# from horovod.tensorflow.compression import Compression
try:
  from horovod.tensorflow.functions import broadcast_variables
except Exception:
  pass

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class BroadcastGlobalVariablesHook(session_run_hook.SessionRunHook):
  """SessionRunHook that will broadcast all global variables from root rank to all other processes during initialization.

  This is necessary to ensure consistent initialization of all workers when
  training is started with random weights or restored from a checkpoint.
  """  # noqa: E501

  def __init__(self, root_rank, device=''):
    """Construct a new BroadcastGlobalVariablesHook that will broadcast all global variables from root rank to all other processes during initialization.

    Args:
      root_rank:
        Rank that will send data, other ranks will receive data.
      device:
        Device to be used for broadcasting. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
    """  # noqa: E501
    super(BroadcastGlobalVariablesHook, self).__init__()
    self.root_rank = root_rank
    self.bcast_op = None
    self.device = device

  def begin(self):
    bcast_vars = []
    embed_para_vars = ops.get_collection(constant.EmbeddingParallel)
    for x in tf.global_variables():
      # if '/embedding' not in x.name and 'DynamicVariable' not in str(type(x)):
      if x.name not in embed_para_vars:
        bcast_vars.append(x)
        logging.info('will broadcast variable: name=%s shape=%s' %
                     (x.name, x.get_shape()))
    if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
      with tf.device(self.device):
        self.bcast_op = broadcast_variables(bcast_vars, self.root_rank)

  def after_create_session(self, session, coord):
    session.run(self.bcast_op)
