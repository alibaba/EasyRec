# -*- encoding:utf-8 -*-
import logging
import os

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.framework import ops
from tensorflow.python.platform.gfile import GFile
# from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.loader_impl import SavedModelLoader

from easy_rec.python.utils import conditional
from easy_rec.python.utils import constant
from easy_rec.python.utils import embedding_utils
from easy_rec.python.utils import proto_util

EMBEDDING_INITIALIZERS = 'embedding_initializers'


class MetaGraphEditor:

  def __init__(self,
               lookup_lib_path,
               saved_model_dir,
               redis_url=None,
               redis_passwd=None,
               redis_timeout=0,
               redis_cache_names=[],
               oss_path=None,
               oss_endpoint=None,
               oss_ak=None,
               oss_sk=None,
               oss_timeout=0,
               meta_graph_def=None,
               norm_name_to_ids=None,
               incr_update_params=None,
               debug_dir=''):
    self._lookup_op = tf.load_op_library(lookup_lib_path)
    self._debug_dir = debug_dir
    self._verbose = debug_dir != ''
    if saved_model_dir:
      tags = ['serve']
      loader = SavedModelLoader(saved_model_dir)
      saver, _ = loader.load_graph(tf.get_default_graph(), tags, None)
      meta_graph_def = loader.get_meta_graph_def_from_tags(tags)
    else:
      assert meta_graph_def, 'either saved_model_dir or meta_graph_def must be set'
      tf.reset_default_graph()
      from tensorflow.python.framework import meta_graph
      meta_graph.import_scoped_meta_graph_with_return_elements(
          meta_graph_def, clear_devices=True)
      # tf.train.import_meta_graph(meta_graph_def)
    self._meta_graph_version = meta_graph_def.meta_info_def.meta_graph_version
    self._signature_def = meta_graph_def.signature_def[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    if self._verbose:
      debug_out_path = os.path.join(self._debug_dir, 'meta_graph_raw.txt')
      with GFile(debug_out_path, 'w') as fout:
        fout.write(text_format.MessageToString(meta_graph_def, as_utf8=True))
    self._meta_graph_def = meta_graph_def
    self._old_node_num = len(self._meta_graph_def.graph_def.node)
    self._all_graph_nodes = None
    self._all_graph_node_flags = None
    self._restore_tensor_node = None
    self._restore_shard_node = None
    self._restore_all_node = []
    self._lookup_outs = None
    self._feature_names = None
    self._embed_names = None
    self._embed_name_to_ids = norm_name_to_ids
    self._is_cache_from_redis = []
    self._redis_cache_names = redis_cache_names
    self._embed_ids = None
    self._embed_dims = None
    self._embed_sizes = None
    self._embed_combiners = None
    self._redis_url = redis_url
    self._redis_passwd = redis_passwd
    self._redis_timeout = redis_timeout
    self._oss_path = oss_path
    self._oss_endpoint = oss_endpoint
    self._oss_ak = oss_ak
    self._oss_sk = oss_sk
    self._oss_timeout = oss_timeout

    self._incr_update_params = incr_update_params

    # increment update placeholders
    self._embedding_update_inputs = {}
    self._embedding_update_outputs = {}

    self._dense_update_inputs = {}
    self._dense_update_outputs = {}

  @property
  def sparse_update_inputs(self):
    return self._embedding_update_inputs

  @property
  def sparse_update_outputs(self):
    return self._embedding_update_outputs

  @property
  def dense_update_inputs(self):
    return self._dense_update_inputs

  @property
  def dense_update_outputs(self):
    return self._dense_update_outputs

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

  def _get_share_embed_name(self, x, embed_names):
    """Map share embedding tensor names to embed names.

    Args:
      x: string, embedding tensor names, such as:
        input_layer_1/shared_embed_1/field16_shared_embedding
        input_layer_1/shared_embed_2/field17_shared_embedding
        input_layer/shared_embed_wide/field15_shared_embedding
        input_layer/shared_embed_wide_1/field16_shared_embedding
      embed_names: all the optional embedding_names
    Return:
      one element in embed_names, such as:
         input_layer_1/shared_embed
         input_layer_1/shared_embed
         input_layer/shared_embed_wide
         input_layer/shared_embed_wide
    """
    assert x.endswith('_shared_embedding')
    name_toks = x.split('/')
    name_toks = name_toks[:-1]
    tmp = name_toks[-1]
    tmp = tmp.split('_')
    try:
      int(tmp[-1])
      name_toks[-1] = '_'.join(tmp[:-1])
    except Exception:
      pass
    tmp_name = '/'.join(name_toks[1:])
    sel_embed_name = ''
    for embed_name in embed_names:
      tmp_toks = embed_name.split('/')
      tmp_toks = tmp_toks[1:]
      embed_name_sub = '/'.join(tmp_toks)
      if tmp_name == embed_name_sub:
        assert not sel_embed_name, 'confusions encountered: %s %s' % (
            x, ','.join(embed_names))
        sel_embed_name = embed_name
    assert sel_embed_name, '%s not find in shared_embeddings: %s' % (
        tmp_name, ','.join(embed_names))
    return sel_embed_name

  def _find_embed_combiners(self, norm_embed_names):
    """Find embedding lookup combiner methods.

    Args:
       norm_embed_names: normalized embedding names
    Return:
       list: combiner methods for each features: sum, mean, sqrtn
    """
    embed_combiners = {}
    embed_combine_node_cts = {}
    combiner_map = {
        'SparseSegmentSum': 'sum',
        'SparseSegmentMean': 'mean',
        'SparseSegmentSqrtN': 'sqrtn'
    }
    for node in self._meta_graph_def.graph_def.node:
      if node.op in combiner_map:
        norm_name, _ = proto_util.get_norm_embed_name(node.name)
        embed_combiners[norm_name] = combiner_map[node.op]
        embed_combine_node_cts[norm_name] = embed_combine_node_cts.get(
            norm_name, 0) + 1
      elif node.op == 'RealDiv' and len(node.input) == 2:
        # for tag feature with weights, and combiner == mean
        if 'SegmentSum' in node.input[0] and 'SegmentSum' in node.input[1]:
          norm_name, _ = proto_util.get_norm_embed_name(node.name)
          embed_combiners[norm_name] = 'mean'
          embed_combine_node_cts[norm_name] = embed_combine_node_cts.get(
              norm_name, 0) + 1
      elif node.op == 'SegmentSum':
        norm_name, _ = proto_util.get_norm_embed_name(node.name)
        # avoid overwrite RealDiv results
        if norm_name not in embed_combiners:
          embed_combiners[norm_name] = 'sum'
        embed_combine_node_cts[norm_name] = embed_combine_node_cts.get(
            norm_name, 0) + 1
    return [embed_combiners[x] for x in norm_embed_names]

  def _find_lookup_indices_values_shapes(self):
    # use the specific _embedding_weights/SparseReshape to find out
    # lookup inputs: indices, values, dense_shape, weights
    indices = {}
    values = {}
    shapes = {}

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

    for node in self._meta_graph_def.graph_def.node:
      if '_embedding_weights/SparseReshape' in node.name:
        if node.op == 'SparseReshape':
          # embed_name, _ = proto_util.get_norm_embed_name(node.name, self._verbose)
          fea_name, _ = proto_util.get_norm_embed_name(node.name, self._verbose)
          for tmp_input in node.input:
            tmp_shape = _get_output_shape(self._meta_graph_def.graph_def,
                                          tmp_input)
            if '_embedding_weights/Cast' in tmp_input:
              continue
            elif len(tmp_shape.dim) == 2:
              indices[fea_name] = tmp_input
            elif len(tmp_shape.dim) == 1:
              shapes[fea_name] = tmp_input
        elif node.op == 'Identity':
          fea_name, _ = proto_util.get_norm_embed_name(node.name, self._verbose)
          values[fea_name] = node.input[0]
    return indices, values, shapes

  def _find_lookup_weights(self):
    weights = {}
    for node in self._meta_graph_def.graph_def.node:
      if '_weighted_by_' in node.name and 'GatherV2' in node.name:
        has_sparse_reshape = False
        for tmp_input in node.input:
          if 'SparseReshape' in tmp_input:
            has_sparse_reshape = True
        if has_sparse_reshape:
          continue
        if len(node.input) != 3:
          continue
        # try to find nodes with weights
        # input_layer/xxx_weighted_by_yyy_embedding/xxx_weighted_by_yyy_embedding_weights/GatherV2_[0-9]
        # which has three inputs:
        #   input_layer/xxx_weighted_by_yyy_embedding/xxx_weighted_by_yyy_embedding_weights/Reshape_1
        #   DeserializeSparse_1    (this is the weight)
        #   input_layer/xxx_weighted_by_yyy_embedding/xxx_weighted_by_yyy_embedding_weights/GatherV2_4/axis
        fea_name, _ = proto_util.get_norm_embed_name(node.name, self._verbose)
        for tmp_input in node.input:
          if '_weighted_by_' not in tmp_input:
            weights[fea_name] = tmp_input
    return weights

  def _find_embed_names_and_dims(self, norm_embed_names):
    # get embedding dimensions from Variables
    embed_dims = {}
    embed_sizes = {}
    embed_is_kv = {}
    for node in self._meta_graph_def.graph_def.node:
      if 'embedding_weights' in node.name and node.op in [
          'VariableV2', 'KvVarHandleOp'
      ]:
        tmp = node.attr['shape'].shape.dim[-1].size
        tmp2 = 1
        for x in node.attr['shape'].shape.dim[:-1]:
          tmp2 = tmp2 * x.size
        embed_name, _ = proto_util.get_norm_embed_name(node.name, self._verbose)
        assert embed_name is not None,\
            'fail to get_norm_embed_name(%s)' % node.name
        embed_dims[embed_name] = tmp
        embed_sizes[embed_name] = tmp2
        embed_is_kv[embed_name] = 1 if node.op == 'KvVarHandleOp' else 0

    # get all embedding dimensions, note that some embeddings
    # are shared by multiple inputs, so the names should be
    # transformed
    all_embed_dims = []
    all_embed_names = []
    all_embed_sizes = []
    all_embed_is_kv = []
    for x in norm_embed_names:
      if x in embed_dims:
        all_embed_names.append(x)
        all_embed_dims.append(embed_dims[x])
        all_embed_sizes.append(embed_sizes[x])
        all_embed_is_kv.append(embed_is_kv[x])
      elif x.endswith('_shared_embedding'):
        tmp_embed_name = self._get_share_embed_name(x, embed_dims.keys())
        all_embed_names.append(tmp_embed_name)
        all_embed_dims.append(embed_dims[tmp_embed_name])
        all_embed_sizes.append(embed_sizes[tmp_embed_name])
        all_embed_is_kv.append(embed_is_kv[tmp_embed_name])
    return all_embed_names, all_embed_dims, all_embed_sizes, all_embed_is_kv

  def find_lookup_inputs(self):
    logging.info('Extract embedding_lookup inputs')

    indices, values, shapes = self._find_lookup_indices_values_shapes()
    weights = self._find_lookup_weights()

    for fea in shapes.keys():
      logging.info('Lookup Input[%s]: indices=%s values=%s shapes=%s' %
                   (fea, indices[fea], values[fea], shapes[fea]))

    graph = tf.get_default_graph()

    def _get_tensor_by_name(tensor_name):
      if ':' not in tensor_name:
        tensor_name = tensor_name + ':0'
      return graph.get_tensor_by_name(tensor_name)

    lookup_input_values = []
    lookup_input_indices = []
    lookup_input_shapes = []
    lookup_input_weights = []
    for key in values.keys():
      tmp_val, tmp_ind, tmp_shape = values[key], indices[key], shapes[key]
      lookup_input_values.append(_get_tensor_by_name(tmp_val))
      lookup_input_indices.append(_get_tensor_by_name(tmp_ind))
      lookup_input_shapes.append(_get_tensor_by_name(tmp_shape))
      if key in weights:
        tmp_w = weights[key]
        lookup_input_weights.append(_get_tensor_by_name(tmp_w))
      else:
        lookup_input_weights.append([])

    # get embedding combiners
    self._embed_combiners = self._find_embed_combiners(values.keys())

    # get embedding dimensions
    self._embed_names, self._embed_dims, self._embed_sizes, self._embed_is_kv\
        = self._find_embed_names_and_dims(values.keys())

    if not self._embed_name_to_ids:
      embed_name_uniq = list(set(self._embed_names))
      self._embed_name_to_ids = {
          t: tid for tid, t in enumerate(embed_name_uniq)
      }
    self._embed_ids = [
        int(self._embed_name_to_ids[x]) for x in self._embed_names
    ]

    self._is_cache_from_redis = [
        proto_util.is_cache_from_redis(x, self._redis_cache_names)
        for x in self._embed_names
    ]

    # normalized feature names
    self._feature_names = list(values.keys())

    return lookup_input_indices, lookup_input_values, lookup_input_shapes,\
        lookup_input_weights

  def add_lookup_op(self, lookup_input_indices, lookup_input_values,
                    lookup_input_shapes, lookup_input_weights):
    logging.info('add custom lookup operation to lookup embeddings from redis')
    self._lookup_outs = [None for i in range(len(lookup_input_values))]
    for i in range(len(lookup_input_values)):
      if lookup_input_values[i].dtype == tf.int32:
        lookup_input_values[i] = tf.to_int64(lookup_input_values[i])
    for i in range(len(self._lookup_outs)):
      i_1 = i + 1
      self._lookup_outs[i] = self._lookup_op.kv_lookup(
          lookup_input_indices[i:i_1],
          lookup_input_values[i:i_1],
          lookup_input_shapes[i:i_1],
          lookup_input_weights[i:i_1],
          url=self._redis_url,
          password=self._redis_passwd,
          timeout=self._redis_timeout,
          combiners=self._embed_combiners[i:i_1],
          embedding_dims=self._embed_dims[i:i_1],
          embedding_names=self._embed_ids[i:i_1],
          cache=self._is_cache_from_redis,
          version=self._meta_graph_version)[0]

    meta_graph_def = tf.train.export_meta_graph()

    if self._verbose:
      debug_path = os.path.join(self._debug_dir, 'graph_raw.txt')
      with GFile(debug_path, 'w') as fout:
        fout.write(
            text_format.MessageToString(
                self._meta_graph_def.graph_def, as_utf8=True))
    return meta_graph_def

  def add_oss_lookup_op(self, lookup_input_indices, lookup_input_values,
                        lookup_input_shapes, lookup_input_weights):
    logging.info('add custom lookup operation to lookup embeddings from oss')
    place_on_cpu = os.getenv('place_embedding_on_cpu')
    place_on_cpu = eval(place_on_cpu) if place_on_cpu else False
    with conditional(place_on_cpu, ops.device('/CPU:0')):
      for i in range(len(lookup_input_values)):
        if lookup_input_values[i].dtype == tf.int32:
          lookup_input_values[i] = tf.to_int64(lookup_input_values[i])
    # N = len(lookup_input_indices)
    # self._lookup_outs = [ None for _ in range(N) ]
    # for i in range(N):
    #   i_1 = i + 1
    #   self._lookup_outs[i] = self._lookup_op.oss_read_kv(
    #       lookup_input_indices[i:i_1],
    #       lookup_input_values[i:i_1],
    #       lookup_input_shapes[i:i_1],
    #       lookup_input_weights[i:i_1],
    #       osspath=self._oss_path,
    #       endpoint=self._oss_endpoint,
    #       ak=self._oss_ak,
    #       sk=self._oss_sk,
    #       timeout=self._oss_timeout,
    #       combiners=self._embed_combiners[i:i_1],
    #       embedding_dims=self._embed_dims[i:i_1],
    #       embedding_ids=self._embed_ids[i:i_1],
    #       embedding_is_kv=self._embed_is_kv[i:i_1],
    #       shared_name='embedding_lookup_res',
    #       name='embedding_lookup_fused/lookup')[0]
    self._lookup_outs = self._lookup_op.oss_read_kv(
        lookup_input_indices,
        lookup_input_values,
        lookup_input_shapes,
        lookup_input_weights,
        osspath=self._oss_path,
        endpoint=self._oss_endpoint,
        ak=self._oss_ak,
        sk=self._oss_sk,
        timeout=self._oss_timeout,
        combiners=self._embed_combiners,
        embedding_dims=self._embed_dims,
        embedding_ids=self._embed_ids,
        embedding_is_kv=self._embed_is_kv,
        shared_name='embedding_lookup_res',
        name='embedding_lookup_fused/lookup')

    N = np.max([int(x) for x in self._embed_ids]) + 1
    uniq_embed_ids = [x for x in range(N)]
    uniq_embed_dims = [0 for x in range(N)]
    uniq_embed_combiners = ['mean' for x in range(N)]
    uniq_embed_is_kvs = [0 for x in range(N)]
    for embed_id, embed_combiner, embed_is_kv, embed_dim in zip(
        self._embed_ids, self._embed_combiners, self._embed_is_kv,
        self._embed_dims):
      uniq_embed_combiners[embed_id] = embed_combiner
      uniq_embed_is_kvs[embed_id] = embed_is_kv
      uniq_embed_dims[embed_id] = embed_dim

    lookup_init_op = self._lookup_op.oss_init(
        osspath=self._oss_path,
        endpoint=self._oss_endpoint,
        ak=self._oss_ak,
        sk=self._oss_sk,
        combiners=uniq_embed_combiners,
        embedding_dims=uniq_embed_dims,
        embedding_ids=uniq_embed_ids,
        embedding_is_kv=uniq_embed_is_kvs,
        N=N,
        shared_name='embedding_lookup_res',
        name='embedding_lookup_fused/init')

    ops.add_to_collection(EMBEDDING_INITIALIZERS, lookup_init_op)

    if self._incr_update_params is not None:
      # all sparse variables are updated by a single custom operation
      message_ph = tf.placeholder(tf.int8, [None], name='incr_update/message')
      embedding_update = self._lookup_op.embedding_update(
          message=message_ph,
          shared_name='embedding_lookup_res',
          name='embedding_lookup_fused/embedding_update')
      self._embedding_update_inputs['incr_update/sparse/message'] = message_ph
      self._embedding_update_outputs[
          'incr_update/sparse/embedding_update'] = embedding_update

      # dense variables are updated one by one
      dense_name_to_ids = embedding_utils.get_dense_name_to_ids()
      for x in ops.get_collection(constant.DENSE_UPDATE_VARIABLES):
        dense_var_id = dense_name_to_ids[x.op.name]
        dense_input_name = 'incr_update/dense/%d/input' % dense_var_id
        dense_output_name = 'incr_update/dense/%d/output' % dense_var_id
        dense_update_input = tf.placeholder(
            tf.float32, x.get_shape(), name=dense_input_name)
        self._dense_update_inputs[dense_input_name] = dense_update_input
        dense_assign_op = tf.assign(x, dense_update_input)
        self._dense_update_outputs[dense_output_name] = dense_assign_op

    meta_graph_def = tf.train.export_meta_graph()

    if self._verbose:
      debug_path = os.path.join(self._debug_dir, 'graph_raw.txt')
      with GFile(debug_path, 'w') as fout:
        fout.write(
            text_format.MessageToString(
                self._meta_graph_def.graph_def, as_utf8=True))
    return meta_graph_def

  def bytes2str(self, x):
    if bytes == str:
      return x
    else:
      try:
        return x.decode('utf-8')
      except Exception:
        # in case of some special chars in protobuf
        return str(x)

  def clear_meta_graph_embeding(self, meta_graph_def):
    logging.info('clear meta graph embedding_weights')

    def _clear_embedding_in_meta_collect(meta_graph_def, collect_name):
      tmp_vals = [
          x
          for x in meta_graph_def.collection_def[collect_name].bytes_list.value
          if 'embedding_weights' not in self.bytes2str(x)
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
      elif key == 'saved_model_assets':
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
            input_name, _ = proto_util.get_norm_embed_name(
                node.input[i], self._verbose)
            print('REPLACE:' + node.input[i] + '=>' + input_name)
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
    # assert self._restore_tensor_node is not None, 'save/RestoreV2/tensor_names is not found'

    if self._restore_tensor_node:
      drop_ids = []
      for tmp_id, tmp_name in enumerate(
          self._restore_tensor_node.attr['value'].tensor.string_val):
        if 'embedding_weights' in self.bytes2str(tmp_name):
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
      elif node.name.startswith('save/restore_all'):
        self._restore_all_node.append(node)

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
        if '_class' in node.attr and len(node.attr['_class'].list.s) > 0:
          const_name = node.attr['_class'].list.s[0]
          if not isinstance(const_name, str):
            const_name = const_name.decode('utf-8')
          if 'embedding_weights' in const_name:
            self._all_graph_node_flags[tid] = False
      elif 'ReadKvVariableOp' in node.name and node.op == 'ReadKvVariableOp':
        all_kv_drop.append(node.name)
        self._all_graph_node_flags[tid] = False
      elif node.op == 'Assign' and 'save/Assign' in node.name:
        # update node(save/Assign_[0-N])'s input[1] by the position of
        #     node.input[0] in save/RestoreV2/tensor_names
        # the outputs of save/RestoreV2 is connected to save/Assign
        tmp_id = [
            self.bytes2str(x)
            for x in self._restore_tensor_node.attr['value'].tensor.string_val
        ].index(node.input[0])
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
    if self._restore_shard_node:
      for tmp_input in drop_save_assigns:
        self._restore_shard_node.input.remove(tmp_input)
        if self._verbose:
          logging.info('drop restore_shard input: %s' % tmp_input)
    elif len(self._restore_all_node) > 0:
      for tmp_input in drop_save_assigns:
        for tmp_node in self._restore_all_node:
          if tmp_input in tmp_node.input:
            tmp_node.input.remove(tmp_input)
            if self._verbose:
              logging.info('drop %s input: %s' % (tmp_node.name, tmp_input))
              break

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
            if 'embedding_weights' in self.bytes2str(tmp_val)
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
        lookup_input_weights = self.find_lookup_inputs()

    # add lookup op to the graph
    self._meta_graph_def = self.add_lookup_op(lookup_input_indices,
                                              lookup_input_values,
                                              lookup_input_shapes,
                                              lookup_input_weights)

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

    if self._verbose:
      debug_dump_path = os.path.join(self._debug_dir, 'graph.txt')
      with GFile(debug_dump_path, 'w') as fout:
        fout.write(text_format.MessageToString(self.graph_def, as_utf8=True))
      debug_dump_path = os.path.join(self._debug_dir, 'meta_graph.txt')
      with GFile(debug_dump_path, 'w') as fout:
        fout.write(
            text_format.MessageToString(self._meta_graph_def, as_utf8=True))

  def edit_graph_for_oss(self):
    # the main entrance
    lookup_input_indices, lookup_input_values, lookup_input_shapes,\
        lookup_input_weights = self.find_lookup_inputs()

    # add lookup op to the graph
    self._meta_graph_def = self.add_oss_lookup_op(lookup_input_indices,
                                                  lookup_input_values,
                                                  lookup_input_shapes,
                                                  lookup_input_weights)

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

    if self._verbose:
      debug_dump_path = os.path.join(self._debug_dir, 'graph.txt')
      with GFile(debug_dump_path, 'w') as fout:
        fout.write(text_format.MessageToString(self.graph_def, as_utf8=True))
      debug_dump_path = os.path.join(self._debug_dir, 'meta_graph.txt')
      with GFile(debug_dump_path, 'w') as fout:
        fout.write(
            text_format.MessageToString(self._meta_graph_def, as_utf8=True))
