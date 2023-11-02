# -*- encoding:utf-8 -*-

import logging
import os
import sys

import numpy as np
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver

try:
  import horovod.tensorflow as hvd
  from sparse_operation_kit.experiment import raw_ops as dynamic_variable_ops
  from sparse_operation_kit import experiment as sok
except:
  dynamic_variable_ops = None
  sok = None

try:
  from tensorflow.python.framework.load_library import load_op_library
  import easy_rec
  load_embed_lib_path = os.path.join(easy_rec.ops_dir, 'libload_embed.so')
  load_embed_lib = load_op_library(load_embed_lib_path)
except Exception as ex:
  logging.warning('load libload_embed.so failed: %s' % str(ex))


class SaverV2(saver.Saver):

  def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False,
               write_version=saver_pb2.SaverDef.V2,
               pad_step_number=False,
               save_relative_paths=False,
               filename=None):
    self._sok_vars = []
    tf_vars = []
    if sok is None:
      tf_vars = var_list
    else:
      for var in var_list:
        if isinstance(var, sok.DynamicVariable):
          self._sok_vars.append(var)
        else:
          tf_vars.append(var)
    super(SaverV2, self).__init__(
        tf_vars,
        reshape=reshape,
        sharded=sharded,
        max_to_keep=max_to_keep,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        name=name,
        restore_sequentially=restore_sequentially,
        saver_def=saver_def,
        builder=builder,
        defer_build=defer_build,
        allow_empty=allow_empty,
        write_version=write_version,
        pad_step_number=pad_step_number,
        save_relative_paths=save_relative_paths,
        filename=filename)
    self._is_build = False

  def _save_sok_embedding(self, sok_var):
    pass

  def _load_sok_embedding(self, sok_var):

    def _load_key_vals(filename, var_name):
      var_name = var_name.decode('utf-8').replace('/', '__')
      filename = filename.decode('utf-8')
      key_file_pattern = filename + '-sok/embed-' + var_name + '-part-*.keys'
      logging.info('key_file_pattern=%s filename=%s var_name=%s var=%s' %
                   (key_file_pattern, filename, var_name, str(sok_var)))
      key_files = gfile.Glob(key_file_pattern)
      logging.info('key_file_pattern=%s file_num=%d' %
                   (key_file_pattern, len(key_files)))
      all_keys = []
      all_vals = []
      for key_file in key_files:
        with gfile.GFile(key_file, 'rb') as fin:
          tmp_keys = np.frombuffer(fin.read(), dtype=np.int64)
          tmp_ids = tmp_keys % hvd.size()
          tmp_ids = np.where(tmp_ids == hvd.rank())[0]
          if len(tmp_ids) == 0:
            break
          all_keys.append(tmp_keys.take(tmp_ids, axis=0))
          logging.info('tmp_keys.shape=%s %s %s' % (str(
              tmp_keys.shape), str(tmp_ids.shape), str(all_keys[-1].shape)))

        val_file = key_file[:-4] + 'vals'
        with gfile.GFile(val_file, 'rb') as fin:
          tmp_vals = np.frombuffer(
              fin.read(), dtype=np.float32).reshape([-1, sok_var._dimension])
          all_vals.append(tmp_vals.take(tmp_ids, axis=0))
          logging.info('tmp_vals.shape=%s %s %s' % (str(
              tmp_vals.shape), str(tmp_ids.shape), str(all_vals[-1].shape)))

      all_keys = np.concatenate(all_keys, axis=0)
      all_vals = np.concatenate(all_vals, axis=0)

      shuffle_ids = np.array(range(len(all_keys)))
      np.random.shuffle(shuffle_ids)
      all_keys = all_keys.take(shuffle_ids, axis=0)
      all_vals = all_vals.take(shuffle_ids, axis=0)
      print(
          len(all_keys), all_vals.shape, np.min(all_keys), np.max(all_keys),
          np.min(all_vals), np.max(all_vals))
      return all_keys, all_vals

    file_name = ops.get_default_graph().get_tensor_by_name(
        self.saver_def.filename_tensor_name)
    # keys, vals = script_ops.py_func(_load_key_vals, [file_name, sok_var.name],
    #      (dtypes.int64, dtypes.float32))
    keys, vals = load_embed_lib.load_kv_embed(
        task_index=hvd.rank(),
        task_num=hvd.size(),
        embed_dim=sok_var._dimension,
        var_name='embed-' + sok_var.name.replace('/', '__'),
        ckpt_path=file_name)
    with ops.control_dependencies([sok_var._initializer_op]):
      return dynamic_variable_ops.dummy_var_assign(sok_var.handle, keys, vals)

  def build(self):
    if self._is_built:
      return
    super(SaverV2, self).build()
    if self.saver_def.restore_op_name and len(self._sok_vars) > 0:
      # load data from the model
      restore_ops = []
      for sok_var in self._sok_vars:
        restore_ops.append(self._load_sok_embedding(sok_var))
      old_restore_op = ops.get_default_graph().get_operation_by_name(
          self.saver_def.restore_op_name)
      restore_ops.append(old_restore_op)
      restore_op_n = control_flow_ops.group(restore_ops)
      self.saver_def.restore_op_name = restore_op_n.name
    # if self.saver_def.save_tensor_name:
    #   save_sok_ops = []
    #   for sok_var in self._sok_vars:
    #     save_sok_ops.
