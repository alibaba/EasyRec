# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging
import os
import shutil
import sys

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.saved_model import signature_constants

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)


class MetaGraphEditor:

  def __init__(self,
               lookup_lib_path,
               saved_model_dir,
               redis_url,
               redis_passwd,
               redis_timeout,
               verbose=False):
    self._lookup_op = tf.load_op_library(lookup_lib_path)
    self._verbose = verbose
    with tf.Session() as sess:
      meta_graph_def = tf.saved_model.loader.load(sess, ['serve'],
                                                  saved_model_dir)
      self._meta_graph_version = meta_graph_def.meta_info_def.meta_graph_version
      self._signature_def = meta_graph_def.signature_def[
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

      if self._verbose:
        with open('meta_graph_raw.txt', 'w') as fout:
          fout.write(text_format.MessageToString(meta_graph_def, as_utf8=True))
      self._meta_graph_def = meta_graph_def
    self._old_node_num = len(self._meta_graph_def.graph_def.node)
    self._all_graph_nodes = None
    self._all_graph_node_flags = None
    self._restore_tensor_node = None
    self._restore_shard_node = None
    self._lookup_outs = None
    self._feature_names = None
    self._embed_names = None
    self._redis_url = redis_url
    self._redis_passwd = redis_passwd
    self._redis_timeout = redis_timeout

  @property
  def graph_def(self):
    return self._meta_graph_def.graph_def

  @property
  def signature_def(self):
    return self._signature_def

  @property
  def meta_graph_version(self):
    return self._meta_graph_version

  def init_graph_node_clear_flags(self):
    graph_def = self._meta_graph_def.graph_def
    self._all_graph_nodes = [n for n in graph_def.node]
    self._all_graph_node_flags = [True for n in graph_def.node]

  def _get_input_name(self, node_name):
    # input_layer/combo_shared/u_city_shared_embedding/u_city_shared_embedding_weights/SparseReshape
    toks = node_name.split('/')
    if '_shared_embedding/' in node_name:
      for x in toks:
        if x.endswith('_shared_embedding'):
          return x[:x.index('_shared_embedding')]
    else:
      # for not shared embedding
      for tok in toks:
        if tok.endswith('_embedding_weights'):
          return tok[:-len('_embedding_weights')]
    return None

  def find_lookup_inputs(self):
    values = {}
    indices = {}
    shapes = {}
    embed_names = {}

    def _get_output_shape(graph_def, input_name):
      out_id = 0
      if ':' in input_name:
        node_name, out_id = input_name.split(':')
        out_id = int(out_id)
      else:
        node_name = input_name
      for node in graph_def.node:
        if node.name == node_name:
          return node.attr['_output_shapes'].list.shape[out_id]
      return None

    def _get_embed_name(node_name):
      toks = node_name.split('/')
      if '_shared_embedding' in node_name:
        # for shared embedding
        for tid, x in enumerate(toks):
          if x.endswith('_shared_embedding'):
            assert tid > 0
            tmp_name = toks[tid - 1]
            tmp_toks = tmp_name.split('_')
            try:
              int(tmp_toks[-1])
              return '_'.join(tmp_toks[:-1])
            except Exception:
              return '_'.join(tmp_toks)
        assert False, 'failed to get embed name from: %s' % node_name
      else:
        # for not shared embedding
        for tok in toks:
          if tok.endswith('_embedding_weights'):
            return tok.replace('_embedding_weights', '')
      return None

    logging.info('Extract embedding_lookup inputs and embedding dimensions')

    # use the specific _embedding_weights/SparseReshape to find out
    # lookup inputs: indices, values, dense_shape
    for node in self._meta_graph_def.graph_def.node:
      if '_embedding_weights/SparseReshape' in node.name:
        if node.op == 'SparseReshape':
          embed_name = _get_embed_name(node.name)
          fea_name = self._get_input_name(node.name)
          for tmp_input in node.input:
            tmp_shape = _get_output_shape(self._meta_graph_def.graph_def,
                                          tmp_input)
            if '_embedding_weights/Cast' in tmp_input:
              continue
            elif len(tmp_shape.dim) == 2:
              indices[fea_name] = tmp_input
            elif len(tmp_shape.dim) == 1:
              shapes[fea_name] = tmp_input
            embed_names[fea_name] = embed_name
        elif node.op == 'Identity':
          embed_name = _get_embed_name(node.name)
          fea_name = self._get_input_name(node.name)
          embed_names[fea_name] = embed_name
          values[fea_name] = node.input[0]

    for fea in shapes.keys():
      logging.info('Lookup Input: indices=%s values=%s shapes=%s' %
                   (indices[fea], values[fea], shapes[fea]))

    graph = tf.get_default_graph()

    lookup_input_values = []
    lookup_input_indices = []
    lookup_input_shapes = []
    for key in values.keys():
      tmp_val, tmp_ind, tmp_shape = values[key], indices[key], shapes[key]
      if ':' not in tmp_val:
        tmp_val = tmp_val + ':0'
      if ':' not in tmp_ind:
        tmp_ind = tmp_ind + ':0'
      if ':' not in tmp_shape:
        tmp_shape = tmp_shape + ':0'
      lookup_input_values.append(graph.get_tensor_by_name(tmp_val))
      lookup_input_indices.append(graph.get_tensor_by_name(tmp_ind))
      lookup_input_shapes.append(graph.get_tensor_by_name(tmp_shape))
    lookup_input_weights = [[] for x in lookup_input_shapes]

    # get embedding dimensions
    embed_dims = {}

    def _get_embed_var_name(node_name):
      tmp_toks = node_name.split('/')
      for i in range(1, len(tmp_toks)):
        if tmp_toks[i] == 'embedding_weights':
          tmp_name = tmp_toks[i - 1]
          if tmp_name.startswith('emd_'):
            # emd_article_id
            # emd_item_tag_id
            tmp_name = tmp_name[len('emd_'):]
          elif tmp_name.startswith('hist_emd_'):
            # hist_emd_usersequence
            tmp_name = tmp_name[len('hist_emd_'):]
          if tmp_name.endswith('_embedding'):
            # city_embedding
            # login_time_span_embedding
            # prefer_category_hist_ctr_embedding
            return tmp_name[:-len('_embedding')]
          else:
            return tmp_name
      return None

    for node in self._meta_graph_def.graph_def.node:
      if 'embedding_weights' in node.name and node.op in [
          'VariableV2', 'KvVarHandleOp'
      ]:
        tmp = node.attr['shape'].shape.dim[-1].size
        embed_dims[_get_embed_var_name(node.name)] = tmp

    embed_dims = [embed_dims[embed_names[x]] for x in values.keys()]
    self._feature_names = values.keys()
    self._embed_names = [embed_names[x] for x in values.keys()]
    return lookup_input_indices, lookup_input_values, lookup_input_shapes,\
        lookup_input_weights, embed_dims, self._embed_names

  def add_lookup_op(self, lookup_input_indices, lookup_input_values,
                    lookup_input_shapes, lookup_input_weights, embed_dims,
                    embed_names):
    logging.info('add custom lookup operation to lookup embeddings from redis')
    for i in range(len(lookup_input_values)):
      if lookup_input_values[i].dtype == tf.int32:
        lookup_input_values[i] = tf.to_int64(lookup_input_values[i])
    self._lookup_outs = self._lookup_op.kv_lookup(
        lookup_input_indices,
        lookup_input_values,
        lookup_input_shapes,
        lookup_input_weights,
        url=self._redis_url,
        password=self._redis_passwd,
        timeout=self._redis_timeout,
        combiners=['mean'] * len(lookup_input_weights),
        embedding_dims=embed_dims,
        embedding_names=embed_names,
        version=self._meta_graph_version)
    meta_graph_def = tf.train.export_meta_graph()

    if self._verbose:
      with open('graph_raw.txt', 'w') as fout:
        fout.write(
            text_format.MessageToString(
                self._meta_graph_def.graph_def, as_utf8=True))
    return meta_graph_def

  def clear_meta_graph_embeding(self, meta_graph_def):
    logging.info('clear meta graph embedding_weights')

    def _clear_embedding_in_meta_collect(meta_graph_def, collect_name):
      tmp_vals = [
          x
          for x in meta_graph_def.collection_def[collect_name].bytes_list.value
          if 'embedding_weights' not in x
      ]
      meta_graph_def.collection_def[collect_name].bytes_list.ClearField('value')
      for tmp_v in tmp_vals:
        meta_graph_def.collection_def[collect_name].bytes_list.value.append(
            tmp_v)

    _clear_embedding_in_meta_collect(meta_graph_def, 'model_variables')
    _clear_embedding_in_meta_collect(meta_graph_def, 'trainable_variables')
    _clear_embedding_in_meta_collect(meta_graph_def, 'variables')

    # clear Kv(pai embedding variable) ops in meta_info_def.stripped_op_list.op
    kept_ops = [
        x for x in meta_graph_def.meta_info_def.stripped_op_list.op
        if x.name not in [
            'InitializeKvVariableOp', 'KvResourceGather', 'KvResourceImportV2',
            'KvVarHandleOp', 'KvVarIsInitializedOp', 'ReadKvVariableOp'
        ]
    ]
    meta_graph_def.meta_info_def.stripped_op_list.ClearField('op')
    meta_graph_def.meta_info_def.stripped_op_list.op.extend(kept_ops)
    for tmp_op in meta_graph_def.meta_info_def.stripped_op_list.op:
      if tmp_op.name == 'SaveV2':
        for tmp_id, tmp_attr in enumerate(tmp_op.attr):
          if tmp_attr.name == 'has_ev':
            tmp_op.attr.remove(tmp_attr)
            break

  def clear_meta_collect(self, meta_graph_def):
    drop_meta_collects = []
    for key in meta_graph_def.collection_def:
      val = meta_graph_def.collection_def[key]
      if val.HasField('node_list'):
        if 'embedding_weights' in val.node_list.value[
            0] and 'easy_rec' not in val.node_list.value[0]:
          drop_meta_collects.append(key)
        elif 'asset_filepaths' == key and val.node_list.value[
            0] == 'pipeline.config:0':
          # we have to drop this to avoid Load tensorflow model exception:
          # Endpoint "pipeline.config:0" fed more than once.
          drop_meta_collects.append(key)
    for key in drop_meta_collects:
      meta_graph_def.collection_def.pop(key)

  def remove_embedding_weights_and_update_lookup_outputs(self):

    def _should_drop(name):
      if '_embedding_weights' in name:
        if self._verbose:
          logging.info('[SHOULD_DROP] %s' % name)
        return True

    logging.info('remove embedding_weights node in graph_def.node')
    logging.info(
        'and replace the old embedding_lookup outputs with new lookup_op outputs'
    )

    for tid, node in enumerate(self._all_graph_nodes):
      # drop the nodes
      if _should_drop(node.name):
        self._all_graph_node_flags[tid] = False
      else:
        for i in range(len(node.input)):
          if _should_drop(node.input[i]):
            input_name = self._get_input_name(node.input[i])
            input_name = self._lookup_outs[self._feature_names.index(
                input_name)].name
            if input_name.endswith(':0'):
              input_name = input_name.replace(':0', '')
            node.input[i] = input_name

  # drop by ids
  def _drop_by_ids(self, tmp_obj, key, drop_ids):
    keep_vals = [
        x for i, x in enumerate(getattr(tmp_obj, key)) if i not in drop_ids
    ]
    tmp_obj.ClearField(key)
    getattr(tmp_obj, key).extend(keep_vals)

  def clear_save_restore(self):
    """Clear save restore ops.

    save/restore_all need save/restore_shard as input
    save/restore_shard needs save/Assign_[0-N] as input
    save/Assign_[0-N] needs save/RestoreV2 as input
    save/RestoreV2 use save/RestoreV2/tensor_names and save/RestoreV2/shape_and_slices as input
    edit [ save/RestoreV2/tensor_names save/RestoreV2/shape_and_slices save/RestoreV2 save/restore_shard ]
    """
    for tid, node in enumerate(self._all_graph_nodes):
      if not self._all_graph_node_flags[tid]:
        continue
      if node.name == 'save/RestoreV2/tensor_names':
        self._restore_tensor_node = node
        break
    assert self._restore_tensor_node is not None, 'save/RestoreV2/tensor_names is not found'

    drop_ids = []
    for tmp_id, tmp_name in enumerate(
        self._restore_tensor_node.attr['value'].tensor.string_val):
      if 'embedding_weights' in tmp_name:
        drop_ids.append(tmp_id)

    self._drop_by_ids(self._restore_tensor_node.attr['value'].tensor,
                      'string_val', drop_ids)
    keep_node_num = len(
        self._restore_tensor_node.attr['value'].tensor.string_val)
    logging.info(
        'update self._restore_tensor_node: string_val keep_num = %d drop_num = %d'
        % (keep_node_num, len(drop_ids)))
    self._restore_tensor_node.attr['value'].tensor.tensor_shape.dim[
        0].size = keep_node_num
    self._restore_tensor_node.attr['_output_shapes'].list.shape[0].dim[
        0].size = keep_node_num

    logging.info(
        'update save/RestoreV2, drop tensor_shapes, _output_shapes, related to embedding_weights'
    )
    self._restore_shard_node = None
    for node_id, node in enumerate(self._all_graph_nodes):
      if not self._all_graph_node_flags[tid]:
        continue
      if node.name == 'save/RestoreV2/shape_and_slices':
        node.attr['value'].tensor.tensor_shape.dim[0].size = keep_node_num
        node.attr['_output_shapes'].list.shape[0].dim[0].size = keep_node_num
        self._drop_by_ids(node.attr['value'].tensor, 'string_val', drop_ids)
      elif node.name == 'save/RestoreV2':
        self._drop_by_ids(node.attr['_output_shapes'].list, 'shape', drop_ids)
        self._drop_by_ids(node.attr['dtypes'].list, 'type', drop_ids)
      elif node.name == 'save/restore_shard':
        self._restore_shard_node = node

  def clear_save_assign(self):
    logging.info(
        'update save/Assign, drop tensor_shapes, _output_shapes, related to embedding_weights'
    )
    # edit save/Assign
    drop_save_assigns = []
    all_kv_drop = []
    for tid, node in enumerate(self._all_graph_nodes):
      if not self._all_graph_node_flags[tid]:
        continue
      if node.op == 'Assign' and 'save/Assign' in node.name and \
         'embedding_weights' in node.input[0]:
        drop_save_assigns.append('^' + node.name)
        self._all_graph_node_flags[tid] = False
      elif 'embedding_weights/ConcatPartitions/concat' in node.name:
        self._all_graph_node_flags[tid] = False
      elif node.name.endswith('/embedding_weights') and node.op == 'Identity':
        self._all_graph_node_flags[tid] = False
      elif 'save/KvResourceImportV2' in node.name and node.op == 'KvResourceImportV2':
        drop_save_assigns.append('^' + node.name)
        self._all_graph_node_flags[tid] = False
      elif 'KvResourceImportV2' in node.name:
        self._all_graph_node_flags[tid] = False
      elif 'save/Const' in node.name and node.op == 'Const':
        if '_class' in node.attr and 'embedding_weights' in node.attr[
            '_class'].list.s[0]:
          self._all_graph_node_flags[tid] = False
      elif 'ReadKvVariableOp' in node.name and node.op == 'ReadKvVariableOp':
        all_kv_drop.append(node.name)
        self._all_graph_node_flags[tid] = False
      elif node.op == 'Assign' and 'save/Assign' in node.name:
        # update node(save/Assign_[0-N])'s input[1] by the position of
        #     node.input[0] in save/RestoreV2/tensor_names
        # the outputs of save/RestoreV2 is connected to save/Assign
        tmp_id = list(
            self._restore_tensor_node.attr['value'].tensor.string_val).index(
                node.input[0])
        if tmp_id != 0:
          tmp_input2 = 'save/RestoreV2:%d' % tmp_id
        else:
          tmp_input2 = 'save/RestoreV2'
        if tmp_input2 != node.input[1]:
          if self._verbose:
            logging.info("update save/Assign[%s]'s input from %s to %s" %
                         (node.name, node.input[1], tmp_input2))
          node.input[1] = tmp_input2

    # save/restore_all need save/restore_shard as input
    # save/restore_shard needs save/Assign_[0-N] as input
    # save/Assign_[0-N] needs save/RestoreV2 as input
    for tmp_input in drop_save_assigns:
      self._restore_shard_node.input.remove(tmp_input)
      if self._verbose:
        logging.info('drop restore_shard input: %s' % tmp_input)

  def clear_save_v2(self):
    """Clear SaveV2 ops.

    save/Identity need [ save/MergeV2Checkpoints, save/control_dependency ]
    as input. Save/MergeV2Checkpoints need [save/MergeV2Checkpoints/checkpoint_prefixes]
    as input. Save/MergeV2Checkpoints/checkpoint_prefixes need [ save/ShardedFilename,
    save/control_dependency ] as input. save/control_dependency need save/SaveV2 as input.
    save/SaveV2 input: [ save/SaveV2/tensor_names, save/SaveV2/shape_and_slices ]
    edit save/SaveV2  save/SaveV2/shape_and_slices save/SaveV2/tensor_names.
    """
    logging.info('update save/SaveV2 input shape, _output_shapes, tensor_shape')
    save_drop_ids = []
    for tid, node in enumerate(self._all_graph_nodes):
      if not self._all_graph_node_flags[tid]:
        continue
      if node.name == 'save/SaveV2' and node.op == 'SaveV2':
        for tmp_id, tmp_input in enumerate(node.input):
          if '/embedding_weights' in tmp_input:
            save_drop_ids.append(tmp_id)
        diff_num = len(node.input) - len(node.attr['dtypes'].list.type)
        self._drop_by_ids(node, 'input', save_drop_ids)
        save_drop_ids = [x - diff_num for x in save_drop_ids]
        self._drop_by_ids(node.attr['dtypes'].list, 'type', save_drop_ids)
        if 'has_ev' in node.attr:
          del node.attr['has_ev']
    for node in self._all_graph_nodes:
      if node.name == 'save/SaveV2/shape_and_slices' and node.op == 'Const':
        # _output_shapes # size # string_val
        node.attr['_output_shapes'].list.shape[0].dim[0].size -= len(
            save_drop_ids)
        node.attr['value'].tensor.tensor_shape.dim[0].size -= len(save_drop_ids)
        self._drop_by_ids(node.attr['value'].tensor, 'string_val',
                          save_drop_ids)
      elif node.name == 'save/SaveV2/tensor_names':
        # tensor_names may not have the same order as save/SaveV2/shape_and_slices
        tmp_drop_ids = [
            tmp_id for tmp_id, tmp_val in enumerate(
                node.attr['value'].tensor.string_val)
            if 'embedding_weights' in tmp_val
        ]
        # attr['value'].tensor.string_val  # tensor_shape  # size
        assert len(save_drop_ids) == len(save_drop_ids)
        node.attr['_output_shapes'].list.shape[0].dim[0].size -= len(
            tmp_drop_ids)
        node.attr['value'].tensor.tensor_shape.dim[0].size -= len(tmp_drop_ids)
        self._drop_by_ids(node.attr['value'].tensor, 'string_val', tmp_drop_ids)

  def clear_initialize(self):
    """Clear initialization ops.

    */read(Identity) depend on [*(VariableV2)]
    */Assign depend on [*/Initializer/*, *(VariableV2)]
    drop embedding_weights initialization nodes
    */embedding_weights/part_x [,/Assign,/read]
    */embedding_weights/part_1/Initializer/truncated_normal [,/shape,/mean,/stddev,/TruncatedNormal,/mul]
    """
    logging.info('Remove Initialization nodes for embedding_weights')
    for tid, node in enumerate(self._all_graph_nodes):
      if not self._all_graph_node_flags[tid]:
        continue
      if 'embedding_weights' in node.name and 'Initializer' in node.name:
        self._all_graph_node_flags[tid] = False
      elif 'embedding_weights' in node.name and 'Assign' in node.name:
        self._all_graph_node_flags[tid] = False
      elif 'embedding_weights' in node.name and node.op == 'VariableV2':
        self._all_graph_node_flags[tid] = False
      elif 'embedding_weights' in node.name and node.name.endswith(
          '/read') and node.op == 'Identity':
        self._all_graph_node_flags[tid] = False
      elif 'embedding_weights' in node.name and node.op == 'Identity':
        node_toks = node.name.split('/')
        node_tok = node_toks[-1]
        if 'embedding_weights_' in node_tok:
          node_tok = node_tok[len('embedding_weights_'):]
          try:
            int(node_tok)
            self._all_graph_node_flags[tid] = False
          except Exception:
            pass

  def clear_embedding_variable(self):
    # for pai embedding variable, we drop some special nodes
    for tid, node in enumerate(self._all_graph_nodes):
      if not self._all_graph_node_flags[tid]:
        continue
      if node.op in [
          'ReadKvVariableOp', 'KvVarIsInitializedOp', 'KvVarHandleOp'
      ]:
        self._all_graph_node_flags[tid] = False

  # there maybe some nodes depend on the dropped nodes, they are dropped as well
  def drop_dependent_nodes(self):
    drop_names = [
        tmp_node.name
        for tid, tmp_node in enumerate(self._all_graph_nodes)
        if not self._all_graph_node_flags[tid]
    ]
    while True:
      more_drop_names = []
      for tid, tmp_node in enumerate(self._all_graph_nodes):
        if not self._all_graph_node_flags[tid]:
          continue
        if len(tmp_node.input) > 0 and tmp_node.input[0] in drop_names:
          logging.info('drop dependent node: %s depend on %s' %
                       (tmp_node.name, tmp_node.input[0]))
          self._all_graph_node_flags[tid] = False
          more_drop_names.append(tmp_node.name)
      drop_names = more_drop_names
      if not drop_names:
        break

  def edit_graph(self):
    # the main entrance
    lookup_input_indices, lookup_input_values, lookup_input_shapes,\
        lookup_input_weights, embed_dims, embed_names =\
        self.find_lookup_inputs()

    # add lookup op to the graph
    self._meta_graph_def = self.add_lookup_op(lookup_input_indices,
                                              lookup_input_values,
                                              lookup_input_shapes,
                                              lookup_input_weights, embed_dims,
                                              embed_names)

    self.clear_meta_graph_embeding(self._meta_graph_def)

    self.clear_meta_collect(self._meta_graph_def)

    self.init_graph_node_clear_flags()

    self.remove_embedding_weights_and_update_lookup_outputs()

    # save/RestoreV2
    self.clear_save_restore()

    # save/Assign
    self.clear_save_assign()

    # save/SaveV2
    self.clear_save_v2()

    self.clear_initialize()

    self.clear_embedding_variable()

    self.drop_dependent_nodes()

    self._meta_graph_def.graph_def.ClearField('node')
    self._meta_graph_def.graph_def.node.extend([
        x for tid, x in enumerate(self._all_graph_nodes)
        if self._all_graph_node_flags[tid]
    ])

    logging.info('old node number = %d' % self._old_node_num)
    logging.info('node number = %d' % len(self._meta_graph_def.graph_def.node))


if __name__ == '__main__':
  """Replace the default embedding_lookup ops with self defined embedding lookup ops.

  The data are now stored in redis, for lookup, it is to retrieve the
  embedding vectors by {version}_{embed_name}_{embed_id}.
  Example:
    python -m easy_rec.python.tools.edit_lookup_graph
      --saved_model_dir rtp_large_embedding_export/1604304644
      --output_dir ./after_edit_save
      --test_data_path data/test/rtp/xys_cxr_fg_sample_test2_with_lbl.txt
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--saved_model_dir', type=str, default=None, help='saved model dir')
  parser.add_argument('--output_dir', type=str, default=None, help='output dir')
  parser.add_argument(
      '--lookup_lib_path',
      type=str,
      default='libkv_lookup.so',
      help='lookup library path')
  parser.add_argument(
      '--redis_url', type=str, default='127.0.0.1:6379', help='redis url')
  parser.add_argument(
      '--redis_passwd', type=str, default='', help='redis password')
  parser.add_argument('--time_out', type=int, default=1500, help='timeout')
  parser.add_argument(
      '--test_data_path',
      type=str,
      default='data/test/rtp/xys_cxr_fg_sample_test2_with_lbl.txt',
      help='test data path')
  parser.add_argument('--verbose', action='store_true', default=False)

  args = parser.parse_args()
  logging.info('saved_model_dir: %s' % args.saved_model_dir)

  if not os.path.exists(os.path.join(args.saved_model_dir, 'saved_model.pb')):
    logging.error('saved_model.pb does not exist in %s' % args.saved_model_dir)
    sys.exit(1)

  logging.info('output_dir: %s' % args.output_dir)
  logging.info('redis_url: %s' % args.redis_url)
  logging.info('lookup_lib_path: %s' % args.lookup_lib_path)

  if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)

  meta_graph_editor = MetaGraphEditor(args.lookup_lib_path,
                                      args.saved_model_dir, args.redis_url,
                                      args.redis_passwd, args.time_out,
                                      args.verbose)
  meta_graph_editor.edit_graph()

  if args.verbose:
    with open('graph.txt', 'w') as fout:
      fout.write(
          text_format.MessageToString(
              meta_graph_editor.graph_def, as_utf8=True))
    with open('meta_graph.txt', 'w') as fout:
      fout.write(
          text_format.MessageToString(
              meta_graph_editor._meta_graph_def, as_utf8=True))

  # import edit graph
  tf.reset_default_graph()
  saver = tf.train.import_meta_graph(meta_graph_editor._meta_graph_def)
  graph = tf.get_default_graph()
  inputs = meta_graph_editor.signature_def.inputs
  inputs_map = {}
  for name, tensor in inputs.items():
    logging.info('model inputs: %s => %s' % (name, tensor.name))
    inputs_map[name] = graph.get_tensor_by_name(tensor.name)

  outputs = meta_graph_editor.signature_def.outputs
  outputs_map = {}
  for name, tensor in outputs.items():
    logging.info('model outputs: %s => %s' % (name, tensor.name))
    outputs_map[name] = graph.get_tensor_by_name(tensor.name)
  with tf.Session() as sess:
    saver.restore(sess, args.saved_model_dir + '/variables/variables')
    tf.saved_model.simple_save(
        sess, args.output_dir, inputs=inputs_map, outputs=outputs_map)
    # the meta_graph_version could not be passed via existing interfaces
    # so we could only write it by the raw methods
    saved_model = saved_model_pb2.SavedModel()
    with open(os.path.join(args.output_dir, 'saved_model.pb'), 'rb') as fin:
      saved_model.ParseFromString(fin.read())
    saved_model.meta_graphs[
        0].meta_info_def.meta_graph_version = meta_graph_editor.meta_graph_version
    with open(os.path.join(args.output_dir, 'saved_model.pb'), 'wb') as fout:
      fout.write(saved_model.SerializeToString())

    logging.info('save output to %s' % args.output_dir)
    if args.test_data_path != '':
      with open(args.test_data_path, 'r') as fin:
        feature_vals = []
        for line_str in fin:
          line_str = line_str.strip()
          line_toks = line_str.split('')
          line_toks = line_toks[1:]
          feature_vals.append(''.join(line_toks))
          if len(feature_vals) >= 32:
            break
        out_vals = sess.run(
            outputs_map, feed_dict={inputs_map['features']: feature_vals})
        logging.info('test_data probs:' + str(out_vals))
