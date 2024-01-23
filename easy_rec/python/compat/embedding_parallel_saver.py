# -*- encoding:utf-8 -*-

import logging
import os

import numpy as np
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver

from easy_rec.python.utils import constant

try:
  import horovod.tensorflow as hvd
  from sparse_operation_kit.experiment import raw_ops as dynamic_variable_ops
  from easy_rec.python.compat import dynamic_variable
except Exception:
  dynamic_variable_ops = None
  dynamic_variable = None

try:
  from tensorflow.python.framework.load_library import load_op_library
  import easy_rec
  load_embed_lib_path = os.path.join(easy_rec.ops_dir, 'libload_embed.so')
  load_embed_lib = load_op_library(load_embed_lib_path)
except Exception as ex:
  logging.warning('load libload_embed.so failed: %s' % str(ex))
  load_embed_lib = None


def _get_embed_part_id(embed_file):
  embed_file = embed_file.split('/')[-1]
  embed_file = embed_file.split('.')[0]
  embed_id = embed_file.split('-')[-1]
  return int(embed_id)


class EmbeddingParallelSaver(saver.Saver):

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
    self._kv_vars = []
    self._embed_vars = []
    tf_vars = []
    embed_para_vars = ops.get_collection(constant.EmbeddingParallel)
    for var in var_list:
      if dynamic_variable is not None and isinstance(
          var, dynamic_variable.DynamicVariable):
        self._kv_vars.append(var)
      elif var.name in embed_para_vars:
        logging.info('save shard embedding %s part_id=%d part_shape=%s' %
                     (var.name, hvd.rank(), var.get_shape()))
        self._embed_vars.append(var)
      else:
        tf_vars.append(var)
    super(EmbeddingParallelSaver, self).__init__(
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

  def _has_embed_vars(self):
    return (len(self._kv_vars) + len(self._embed_vars)) > 0

  def _save_dense_embedding(self, embed_var):
    logging.info('task[%d] save_dense_embed: %s' % (hvd.rank(), embed_var.name))

    def _save_embed(embed, filename, var_name):
      task_id = hvd.rank()
      filename = filename.decode('utf-8')
      var_name = var_name.decode('utf-8').replace('/', '__')
      embed_dir = filename + '-embedding/'
      logging.info('task[%d] save_dense_embed: %s to %s' %
                   (task_id, var_name, embed_dir))
      if not gfile.Exists(embed_dir):
        gfile.MakeDirs(embed_dir)
      embed_file = filename + '-embedding/embed-' + var_name + '-part-%d.bin' % task_id
      with gfile.GFile(embed_file, 'wb') as fout:
        fout.write(embed.tobytes())

      if task_id == 0:
        # clear old embedding tables
        embed_pattern = filename + '-embedding/embed-' + var_name + '-part-*.bin'
        embed_files = gfile.Glob(embed_pattern)
        for embed_file in embed_files:
          embed_id = _get_embed_part_id(embed_file)
          if embed_id >= hvd.size():
            gfile.DeleteRecursively(embed_file)
      return np.asarray([embed_file], order='C', dtype=np.object)

    file_name = ops.get_default_graph().get_tensor_by_name(
        self.saver_def.filename_tensor_name)
    save_paths = script_ops.py_func(_save_embed,
                                    [embed_var, file_name, embed_var.name],
                                    dtypes.string)
    return save_paths

  def _load_dense_embedding(self, embed_var):
    file_name = ops.get_default_graph().get_tensor_by_name(
        self.saver_def.filename_tensor_name)
    embed_dim = embed_var.get_shape()[-1]
    embed_part_size = embed_var.get_shape()[0]

    def _load_embed(embed, embed_dim, embed_part_size, part_id, part_num,
                    filename, var_name):
      filename = filename.decode('utf-8')
      var_name = var_name.decode('utf-8').replace('/', '__')
      embed_pattern = filename + '-embedding/embed-' + var_name + '-part-*.bin'
      embed_files = gfile.Glob(embed_pattern)

      embed_files.sort(key=_get_embed_part_id)

      logging.info('task[%d] embed_files=%s embed_dim=%d embed_part_size=%d' %
                   (part_id, ','.join(embed_files), embed_dim, embed_part_size))

      part_embed_vals = np.zeros([embed_part_size, embed_dim], dtype=np.float32)
      part_update_cnt = 0
      for embed_file in embed_files:
        part_id_o = _get_embed_part_id(embed_file)
        with gfile.GFile(embed_file, 'rb') as fin:
          embed_val = np.frombuffer(fin.read(), np.float32)
          embed_val = embed_val.reshape([-1, embed_dim])
          embed_ids_o = np.arange(len(embed_val))
          embed_ids_o = part_id_o + embed_ids_o * len(embed_files)
          sel_ids = np.where(
              np.logical_and((embed_ids_o % part_num) == part_id,
                             embed_ids_o < embed_part_size * part_num))[0]
          part_update_cnt += len(sel_ids)
          embed_ids = embed_ids_o[sel_ids]
          embed_ids_n = np.array(embed_ids / part_num, dtype=np.int64)
          part_embed_vals[embed_ids_n] = embed_val[sel_ids]
      logging.info('task[%d] load_part_cnt=%d' % (part_id, part_update_cnt))
      return part_embed_vals

    with ops.control_dependencies([embed_var._initializer_op]):
      if load_embed_lib is not None:
        embed_val = load_embed_lib.load_embed(
            task_index=hvd.rank(),
            task_num=hvd.size(),
            embed_dim=embed_dim,
            embed_part_size=embed_part_size,
            var_name='embed-' + embed_var.name.replace('/', '__'),
            ckpt_path=file_name)
      else:
        embed_val = script_ops.py_func(_load_embed, [
            embed_var, embed_dim, embed_part_size,
            hvd.rank(),
            hvd.size(), file_name, embed_var.name
        ], dtypes.float32)
      embed_val.set_shape(embed_var.get_shape())
      return state_ops.assign(embed_var, embed_val)

  def _save_kv_embedding(self, sok_var):
    indices, values = dynamic_variable_ops.dummy_var_export(
        sok_var.handle, key_type=sok_var.key_type, dtype=sok_var.handle_dtype)
    file_name = ops.get_default_graph().get_tensor_by_name(
        self.saver_def.filename_tensor_name)

    def _save_key_vals(indices, values, filename, var_name):
      var_name = var_name.decode('utf-8').replace('/', '__')
      filename = filename.decode('utf-8')
      sok_dir = filename + '-embedding/'
      if not gfile.Exists(sok_dir):
        gfile.MakeDirs(sok_dir)
      task_id = hvd.rank()
      key_file = filename + '-embedding/embed-' + var_name + '-part-%d.key' % task_id
      with gfile.GFile(key_file, 'wb') as fout:
        fout.write(indices.tobytes())
      val_file = filename + '-embedding/embed-' + var_name + '-part-%d.val' % task_id
      with gfile.GFile(val_file, 'wb') as fout:
        fout.write(values.tobytes())

      if task_id == 0:
        key_file_pattern = filename + '-embedding/embed-' + var_name + '-part-*.key'
        key_files = gfile.Glob(key_file_pattern)
        for key_file in key_files:
          embed_id = _get_embed_part_id(key_file)
          if embed_id >= hvd.size():
            gfile.DeleteRecursively(key_file)
            val_file = key_file[:-4] + '.val'
            if gfile.Exists(val_file):
              gfile.DeleteRecursively(val_file)

      return np.asarray([key_file, val_file], order='C', dtype=np.object)

    save_paths = script_ops.py_func(_save_key_vals,
                                    [indices, values, file_name, sok_var.name],
                                    dtypes.string)
    return save_paths

  def _load_kv_embedding(self, sok_var):

    def _load_key_vals(filename, var_name):
      var_name = var_name.decode('utf-8').replace('/', '__')
      filename = filename.decode('utf-8')
      key_file_pattern = filename + '-embedding/embed-' + var_name + '-part-*.key'
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
          logging.info('part_keys.shape=%s %s %s' % (str(
              tmp_keys.shape), str(tmp_ids.shape), str(all_keys[-1].shape)))

        val_file = key_file[:-4] + 'vals'
        with gfile.GFile(val_file, 'rb') as fin:
          tmp_vals = np.frombuffer(
              fin.read(), dtype=np.float32).reshape([-1, sok_var._dimension])
          all_vals.append(tmp_vals.take(tmp_ids, axis=0))
          logging.info('part_vals.shape=%s %s %s' % (str(
              tmp_vals.shape), str(tmp_ids.shape), str(all_vals[-1].shape)))

      all_keys = np.concatenate(all_keys, axis=0)
      all_vals = np.concatenate(all_vals, axis=0)

      shuffle_ids = np.array(range(len(all_keys)))
      np.random.shuffle(shuffle_ids)
      all_keys = all_keys.take(shuffle_ids, axis=0)
      all_vals = all_vals.take(shuffle_ids, axis=0)
      return all_keys, all_vals

    file_name = ops.get_default_graph().get_tensor_by_name(
        self.saver_def.filename_tensor_name)
    if load_embed_lib is not None:
      keys, vals = load_embed_lib.load_kv_embed(
          task_index=hvd.rank(),
          task_num=hvd.size(),
          embed_dim=sok_var._dimension,
          var_name='embed-' + sok_var.name.replace('/', '__'),
          ckpt_path=file_name)
    else:
      logging.warning('libload_embed.so not loaded, will use python script_ops')
      keys, vals = script_ops.py_func(_load_key_vals, [file_name, sok_var.name],
                                      (dtypes.int64, dtypes.float32))
    with ops.control_dependencies([sok_var._initializer_op]):
      return dynamic_variable_ops.dummy_var_assign(sok_var.handle, keys, vals)

  def build(self):
    if self._is_built:
      return
    super(EmbeddingParallelSaver, self).build()
    if self.saver_def.restore_op_name and self._has_embed_vars():
      # load data from the model
      restore_ops = []
      for sok_var in self._kv_vars:
        restore_ops.append(self._load_kv_embedding(sok_var))
      for embed_var in self._embed_vars:
        restore_ops.append(self._load_dense_embedding(embed_var))
      old_restore_op = ops.get_default_graph().get_operation_by_name(
          self.saver_def.restore_op_name)
      restore_ops.append(old_restore_op)
      restore_op_n = control_flow_ops.group(restore_ops)
      self.saver_def.restore_op_name = restore_op_n.name

    if self.saver_def.save_tensor_name and self._has_embed_vars():
      file_name = ops.get_default_graph().get_tensor_by_name(
          self.saver_def.filename_tensor_name)
      save_part_ops = []
      for sok_var in self._kv_vars:
        save_part_op = self._save_kv_embedding(sok_var)
        save_part_ops.append(save_part_op)
      for embed_var in self._embed_vars:
        save_part_op = self._save_dense_embedding(embed_var)
        save_part_ops.append(save_part_op)
      old_save_op = ops.get_default_graph().get_tensor_by_name(
          self.saver_def.save_tensor_name)
      # only the first worker needs to save non embedding variables
      if hvd.rank() == 0:
        save_part_ops.append(old_save_op)
      with ops.control_dependencies(save_part_ops):
        save_op_n = array_ops.identity(file_name)
      self.saver_def.save_tensor_name = save_op_n.name
