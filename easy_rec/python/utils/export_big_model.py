# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import os
import time

import numpy as np
import tensorflow as tf
from google.protobuf import json_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops.variables import global_variables
from tensorflow.python.platform.gfile import DeleteRecursively
from tensorflow.python.platform.gfile import Exists
from tensorflow.python.platform.gfile import GFile
from tensorflow.python.platform.gfile import Remove
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.monitored_session import ChiefSessionCreator
from tensorflow.python.training.monitored_session import Scaffold
from tensorflow.python.training.saver import export_meta_graph

import easy_rec
from easy_rec.python.utils import constant
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import io_util
from easy_rec.python.utils import proto_util
from easy_rec.python.utils.meta_graph_editor import EMBEDDING_INITIALIZERS
from easy_rec.python.utils.meta_graph_editor import MetaGraphEditor

if tf.__version__ >= '2.0':
  from tensorflow.python.framework.ops import disable_eager_execution
  disable_eager_execution()

ConfigProto = config_pb2.ConfigProto
GPUOptions = config_pb2.GPUOptions

INCR_UPDATE_SIGNATURE_KEY = 'incr_update_sig'


def export_big_model(export_dir, pipeline_config, redis_params,
                     serving_input_fn, estimator, checkpoint_path, verbose):
  for key in redis_params:
    logging.info('%s: %s' % (key, redis_params[key]))

  redis_cache_names = []
  for feature_config in pipeline_config.feature_configs:
    if feature_config.is_cache:
      if feature_config.feature_name:
        redis_cache_names.append(feature_config.feature_name)
      else:
        redis_cache_names.append(feature_config.input_names[0])
  logging.info('The list of cache names: %s' % ','.join(redis_cache_names))

  write_kv_lib_path = os.path.join(easy_rec.ops_dir, 'libembed_op.so')
  kv_module = tf.load_op_library(write_kv_lib_path)

  try:
    sparse_kv_lib_path = os.path.join(easy_rec.ops_dir, 'libwrite_sparse_kv.so')
    sparse_kv_module = tf.load_op_library(sparse_kv_lib_path)
  except Exception as ex:
    logging.warning('load libwrite_sparse_kv.so failed: %s' % str(ex))
    sparse_kv_module = None
  if not checkpoint_path:
    checkpoint_path = estimator_utils.latest_checkpoint(
        pipeline_config.model_dir)
  logging.info('checkpoint_path = %s' % checkpoint_path)

  server = None
  cluster = None
  if 'TF_CONFIG' in os.environ:
    # change chief to master
    tf_config = estimator_utils.chief_to_master()
    if tf_config['task']['type'] == 'ps':
      cluster = tf.train.ClusterSpec(tf_config['cluster'])
      server = tf.train.Server(
          cluster, job_name='ps', task_index=tf_config['task']['index'])
      server.join()
    elif tf_config['task']['type'] == 'master':
      if 'ps' in tf_config['cluster']:
        cluster = tf.train.ClusterSpec(tf_config['cluster'])
        server = tf.train.Server(cluster, job_name='master', task_index=0)
        server_target = server.target
        logging.info('server_target = %s' % server_target)

  serving_input = serving_input_fn()
  features = serving_input.features
  inputs = serving_input.receiver_tensors

  if cluster:
    logging.info('cluster = ' + str(cluster))
  with tf.device(
      replica_device_setter(
          worker_device='/job:master/task:0', cluster=cluster)):
    outputs = estimator._export_model_fn(features, None, None,
                                         estimator.params).predictions

  meta_graph_def = export_meta_graph()
  redis_embedding_version = redis_params.get('redis_embedding_version', '')
  if not redis_embedding_version:
    meta_graph_def.meta_info_def.meta_graph_version =\
        str(int(time.time()))
  else:
    meta_graph_def.meta_info_def.meta_graph_version = redis_embedding_version

  logging.info('meta_graph_version = %s' %
               meta_graph_def.meta_info_def.meta_graph_version)

  embed_var_parts = {}
  embed_norm_name = {}
  embed_spos = {}
  # pai embedding variable
  embedding_vars = {}
  norm_name_to_ids = {}
  for x in global_variables():
    if 'EmbeddingVariable' in str(type(x)):
      norm_name, part_id = proto_util.get_norm_embed_name(x.name)
      norm_name_to_ids[norm_name] = 1
      tmp_export = x.export()
      if x.device not in embedding_vars:
        embedding_vars[x.device] = [(norm_name, tmp_export.keys,
                                     tmp_export.values)]
      else:
        embedding_vars[x.device].append(
            (norm_name, tmp_export.keys, tmp_export.values))
    elif '/embedding_weights:' in x.name or '/embedding_weights/part_' in x.name:
      norm_name, part_id = proto_util.get_norm_embed_name(x.name)
      norm_name_to_ids[norm_name] = 1
      embed_norm_name[x] = norm_name
      if norm_name not in embed_var_parts:
        embed_var_parts[norm_name] = {part_id: x}
      else:
        embed_var_parts[norm_name][part_id] = x

  for tid, t in enumerate(norm_name_to_ids.keys()):
    norm_name_to_ids[t] = str(tid)

  is_cache_from_redis = [  # noqa: F841
      proto_util.is_cache_from_redis(x, redis_cache_names)
      for x in norm_name_to_ids
  ]

  for x in embed_norm_name:
    embed_norm_name[x] = norm_name_to_ids[embed_norm_name[x]]

  total_num = 0
  for norm_name in embed_var_parts:
    parts = embed_var_parts[norm_name]
    spos = 0
    part_ids = list(parts.keys())
    part_ids.sort()
    total_num += len(part_ids)
    for part_id in part_ids:
      embed_spos[parts[part_id]] = spos
      spos += parts[part_id].get_shape()[0]

  redis_url = redis_params.get('redis_url', '')
  redis_passwd = redis_params.get('redis_passwd', '')
  logging.info('will export to redis: %s %s' % (redis_url, redis_passwd))

  if redis_params.get('redis_write_kv', ''):
    # group embed by devices
    per_device_vars = {}
    for x in embed_norm_name:
      if x.device not in per_device_vars:
        per_device_vars[x.device] = [x]
      else:
        per_device_vars[x.device].append(x)

    all_write_res = []
    for tmp_dev in per_device_vars:
      tmp_vars = per_device_vars[tmp_dev]
      with tf.device(tmp_dev):
        tmp_names = [embed_norm_name[v] for v in tmp_vars]
        tmp_spos = [np.array(embed_spos[v], dtype=np.int64) for v in tmp_vars]
        write_kv_res = kv_module.write_kv(
            tmp_names,
            tmp_vars,
            tmp_spos,
            url=redis_url,
            password=redis_passwd,
            timeout=redis_params.get('redis_timeout', 1500),
            version=meta_graph_def.meta_info_def.meta_graph_version,
            threads=redis_params.get('redis_threads', 5),
            batch_size=redis_params.get('redis_batch_size', 32),
            expire=redis_params.get('redis_expire', 24),
            verbose=verbose)
        all_write_res.append(write_kv_res)

    for tmp_dev in embedding_vars:
      with tf.device(tmp_dev):
        tmp_vs = embedding_vars[tmp_dev]
        tmp_sparse_names = [norm_name_to_ids[x[0]] for x in tmp_vs]
        tmp_sparse_keys = [x[1] for x in tmp_vs]
        tmp_sparse_vals = [x[2] for x in tmp_vs]
        write_sparse_kv_res = sparse_kv_module.write_sparse_kv(
            tmp_sparse_names,
            tmp_sparse_vals,
            tmp_sparse_keys,
            url=redis_url,
            password=redis_passwd,
            timeout=redis_params.get('redis_timeout', 1500),
            version=meta_graph_def.meta_info_def.meta_graph_version,
            threads=redis_params.get('redis_threads', 5),
            batch_size=redis_params.get('redis_batch_size', 32),
            expire=redis_params.get('redis_expire', 24),
            verbose=verbose)
        all_write_res.append(write_sparse_kv_res)

    session_config = ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    chief_sess_creator = ChiefSessionCreator(
        master=server.target if server else '',
        checkpoint_filename_with_path=checkpoint_path,
        config=session_config)
    with tf.train.MonitoredSession(
        session_creator=chief_sess_creator,
        hooks=None,
        stop_grace_period_secs=120) as sess:
      dump_flags = sess.run(all_write_res)
      logging.info('write embedding to redis succeed: %s' % str(dump_flags))
  else:
    logging.info('will skip write embedding to redis because '
                 'redis_write_kv is set to 0.')

  # delete embedding_weights collections so that it could be re imported
  tmp_drop = []
  for k in meta_graph_def.collection_def:
    v = meta_graph_def.collection_def[k]
    if len(
        v.node_list.value) > 0 and 'embedding_weights' in v.node_list.value[0]:
      tmp_drop.append(k)
  for k in tmp_drop:
    meta_graph_def.collection_def.pop(k)

  meta_graph_editor = MetaGraphEditor(
      os.path.join(easy_rec.ops_dir, 'libembed_op.so'),
      None,
      redis_url,
      redis_passwd,
      redis_timeout=redis_params.get('redis_timeout', 600),
      redis_cache_names=redis_cache_names,
      meta_graph_def=meta_graph_def,
      norm_name_to_ids=norm_name_to_ids,
      debug_dir=export_dir if verbose else '')
  meta_graph_editor.edit_graph()
  tf.reset_default_graph()

  saver = tf.train.import_meta_graph(meta_graph_editor._meta_graph_def)
  graph = tf.get_default_graph()

  embed_name_to_id_file = os.path.join(export_dir, 'embed_name_to_ids.txt')
  with GFile(embed_name_to_id_file, 'w') as fout:
    for tmp_norm_name in norm_name_to_ids:
      fout.write('%s\t%s\n' % (tmp_norm_name, norm_name_to_ids[tmp_norm_name]))
  ops.add_to_collection(
      tf.GraphKeys.ASSET_FILEPATHS,
      tf.constant(
          embed_name_to_id_file, dtype=tf.string, name='embed_name_to_ids.txt'))

  export_dir = os.path.join(export_dir,
                            meta_graph_def.meta_info_def.meta_graph_version)
  export_dir = io_util.fix_oss_dir(export_dir)
  logging.info('export_dir=%s' % export_dir)
  if Exists(export_dir):
    logging.info('will delete old dir: %s' % export_dir)
    DeleteRecursively(export_dir)

  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  tensor_info_inputs = {}
  for tmp_key in inputs:
    tmp = graph.get_tensor_by_name(inputs[tmp_key].name)
    tensor_info_inputs[tmp_key] = \
        tf.saved_model.utils.build_tensor_info(tmp)
  tensor_info_outputs = {}
  for tmp_key in outputs:
    tmp = graph.get_tensor_by_name(outputs[tmp_key].name)
    tensor_info_outputs[tmp_key] = \
        tf.saved_model.utils.build_tensor_info(tmp)
  signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs=tensor_info_inputs,
          outputs=tensor_info_outputs,
          method_name=signature_constants.PREDICT_METHOD_NAME))

  session_config = ConfigProto(
      allow_soft_placement=True, log_device_placement=True)

  saver = tf.train.Saver()
  with tf.Session(target=server.target if server else '') as sess:
    saver.restore(sess, checkpoint_path)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
        },
        assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        saver=saver,
        strip_default_attrs=True,
        clear_devices=True)
    builder.save()

  # remove temporary files
  Remove(embed_name_to_id_file)
  return export_dir


def export_big_model_to_oss(export_dir, pipeline_config, oss_params,
                            serving_input_fn, estimator, checkpoint_path,
                            verbose):
  for key in oss_params:
    logging.info('%s: %s' % (key, oss_params[key]))

  write_kv_lib_path = os.path.join(easy_rec.ops_dir, 'libembed_op.so')
  kv_module = tf.load_op_library(write_kv_lib_path)

  if not checkpoint_path:
    checkpoint_path = estimator_utils.latest_checkpoint(
        pipeline_config.model_dir)
  logging.info('checkpoint_path = %s' % checkpoint_path)

  server = None
  cluster = None
  if 'TF_CONFIG' in os.environ:
    # change chief to master
    tf_config = estimator_utils.chief_to_master()
    if tf_config['task']['type'] == 'ps':
      cluster = tf.train.ClusterSpec(tf_config['cluster'])
      server = tf.train.Server(
          cluster, job_name='ps', task_index=tf_config['task']['index'])
      server.join()
    elif tf_config['task']['type'] == 'master':
      if 'ps' in tf_config['cluster']:
        cluster = tf.train.ClusterSpec(tf_config['cluster'])
        server = tf.train.Server(cluster, job_name='master', task_index=0)
        server_target = server.target
        logging.info('server_target = %s' % server_target)

  serving_input = serving_input_fn()
  features = serving_input.features
  inputs = serving_input.receiver_tensors

  if cluster:
    logging.info('cluster = ' + str(cluster))
  with tf.device(
      replica_device_setter(
          worker_device='/job:master/task:0', cluster=cluster)):
    outputs = estimator._export_model_fn(features, None, None,
                                         estimator.params).predictions

  meta_graph_def = export_meta_graph()
  meta_graph_def.meta_info_def.meta_graph_version = str(int(time.time()))
  oss_embedding_version = oss_params.get('oss_embedding_version', '')
  if not oss_embedding_version:
    meta_graph_def.meta_info_def.meta_graph_version =\
        str(int(time.time()))
  else:
    meta_graph_def.meta_info_def.meta_graph_version = oss_embedding_version

  logging.info('meta_graph_version = %s' %
               meta_graph_def.meta_info_def.meta_graph_version)

  embed_var_parts = {}
  embed_norm_name = {}
  embed_spos = {}
  # pai embedding variable
  embedding_vars = {}
  norm_name_to_ids = {}
  for x in global_variables():
    tf.logging.info('global var: %s %s %s' % (x.name, str(type(x)), x.device))
    if 'EmbeddingVariable' in str(type(x)):
      norm_name, part_id = proto_util.get_norm_embed_name(x.name)
      norm_name_to_ids[norm_name] = 1
      tmp_export = x.export()
      if x.device not in embedding_vars:
        embedding_vars[x.device] = [(norm_name, tmp_export.keys,
                                     tmp_export.values, part_id)]
      else:
        embedding_vars[x.device].append(
            (norm_name, tmp_export.keys, tmp_export.values, part_id))
    elif '/embedding_weights:' in x.name or '/embedding_weights/part_' in x.name:
      norm_name, part_id = proto_util.get_norm_embed_name(x.name)
      norm_name_to_ids[norm_name] = 1
      embed_norm_name[x] = norm_name
      if norm_name not in embed_var_parts:
        embed_var_parts[norm_name] = {part_id: x}
      else:
        embed_var_parts[norm_name][part_id] = x

  for tid, t in enumerate(norm_name_to_ids.keys()):
    norm_name_to_ids[t] = str(tid)

  for x in embed_norm_name:
    embed_norm_name[x] = norm_name_to_ids[embed_norm_name[x]]

  total_num = 0
  for norm_name in embed_var_parts:
    parts = embed_var_parts[norm_name]
    spos = 0
    part_ids = list(parts.keys())
    part_ids.sort()
    total_num += len(part_ids)
    for part_id in part_ids:
      embed_spos[parts[part_id]] = spos
      spos += parts[part_id].get_shape()[0]

  oss_path = oss_params.get('oss_path', '')
  oss_endpoint = oss_params.get('oss_endpoint', '')
  oss_ak = oss_params.get('oss_ak', '')
  oss_sk = oss_params.get('oss_sk', '')
  logging.info('will export to oss: %s %s %s %s', oss_path, oss_endpoint,
               oss_ak, oss_sk)

  if oss_params.get('oss_write_kv', ''):
    # group embed by devices
    per_device_vars = {}
    for x in embed_norm_name:
      if x.device not in per_device_vars:
        per_device_vars[x.device] = [x]
      else:
        per_device_vars[x.device].append(x)

    all_write_res = []
    for tmp_dev in per_device_vars:
      tmp_vars = per_device_vars[tmp_dev]
      with tf.device(tmp_dev):
        tmp_names = [embed_norm_name[v] for v in tmp_vars]
        tmp_spos = [np.array(embed_spos[v], dtype=np.int64) for v in tmp_vars]
        write_kv_res = kv_module.oss_write_kv(
            tmp_names,
            tmp_vars,
            tmp_spos,
            osspath=oss_path,
            endpoint=oss_endpoint,
            ak=oss_ak,
            sk=oss_sk,
            threads=oss_params.get('oss_threads', 5),
            timeout=5,
            expire=5,
            verbose=verbose)
        all_write_res.append(write_kv_res)

    for tmp_dev in embedding_vars:
      with tf.device(tmp_dev):
        tmp_vs = embedding_vars[tmp_dev]
        tmp_sparse_names = [norm_name_to_ids[x[0]] for x in tmp_vs]
        tmp_sparse_keys = [x[1] for x in tmp_vs]
        tmp_sparse_vals = [x[2] for x in tmp_vs]
        tmp_part_ids = [x[3] for x in tmp_vs]
        write_sparse_kv_res = kv_module.oss_write_sparse_kv(
            tmp_sparse_names,
            tmp_sparse_vals,
            tmp_sparse_keys,
            tmp_part_ids,
            osspath=oss_path,
            endpoint=oss_endpoint,
            ak=oss_ak,
            sk=oss_sk,
            version=meta_graph_def.meta_info_def.meta_graph_version,
            threads=oss_params.get('oss_threads', 5),
            verbose=verbose)
        all_write_res.append(write_sparse_kv_res)

    session_config = ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    chief_sess_creator = ChiefSessionCreator(
        master=server.target if server else '',
        checkpoint_filename_with_path=checkpoint_path,
        config=session_config)
    with tf.train.MonitoredSession(
        session_creator=chief_sess_creator,
        hooks=None,
        stop_grace_period_secs=120) as sess:
      dump_flags = sess.run(all_write_res)
      logging.info('write embedding to oss succeed: %s' % str(dump_flags))
  else:
    logging.info('will skip write embedding to oss because '
                 'oss_write_kv is set to 0.')

  # delete embedding_weights collections so that it could be re imported
  tmp_drop = []
  for k in meta_graph_def.collection_def:
    v = meta_graph_def.collection_def[k]
    if len(
        v.node_list.value) > 0 and 'embedding_weights' in v.node_list.value[0]:
      tmp_drop.append(k)
  for k in tmp_drop:
    meta_graph_def.collection_def.pop(k)

  meta_graph_editor = MetaGraphEditor(
      os.path.join(easy_rec.ops_dir, 'libembed_op.so'),
      None,
      oss_path=oss_path,
      oss_endpoint=oss_endpoint,
      oss_ak=oss_ak,
      oss_sk=oss_sk,
      oss_timeout=oss_params.get('oss_timeout', 1500),
      meta_graph_def=meta_graph_def,
      norm_name_to_ids=norm_name_to_ids,
      incr_update_params=oss_params.get('incr_update', None),
      debug_dir=export_dir if verbose else '')
  meta_graph_editor.edit_graph_for_oss()
  tf.reset_default_graph()

  saver = tf.train.import_meta_graph(meta_graph_editor._meta_graph_def)
  graph = tf.get_default_graph()

  embed_name_to_id_file = os.path.join(export_dir, 'embed_name_to_ids.txt')
  with GFile(embed_name_to_id_file, 'w') as fout:
    for tmp_norm_name in norm_name_to_ids:
      fout.write('%s\t%s\n' % (tmp_norm_name, norm_name_to_ids[tmp_norm_name]))
  ops.add_to_collection(
      ops.GraphKeys.ASSET_FILEPATHS,
      tf.constant(
          embed_name_to_id_file, dtype=tf.string, name='embed_name_to_ids.txt'))

  if 'incr_update' in oss_params:
    dense_train_vars_path = os.path.join(
        os.path.dirname(checkpoint_path), constant.DENSE_UPDATE_VARIABLES)
    ops.add_to_collection(
        ops.GraphKeys.ASSET_FILEPATHS,
        tf.constant(
            dense_train_vars_path,
            dtype=tf.string,
            name=constant.DENSE_UPDATE_VARIABLES))

    asset_file = 'incr_update.txt'
    asset_file_path = os.path.join(export_dir, asset_file)
    with GFile(asset_file_path, 'w') as fout:
      incr_update = oss_params['incr_update']
      incr_update_json = {}
      if 'kafka' in incr_update:
        incr_update_json['storage'] = 'kafka'
        incr_update_json['kafka'] = json.loads(
            json_format.MessageToJson(
                incr_update['kafka'], preserving_proto_field_name=True))
      elif 'datahub' in incr_update:
        incr_update_json['storage'] = 'datahub'
        incr_update_json['datahub'] = json.loads(
            json_format.MessageToJson(
                incr_update['datahub'], preserving_proto_field_name=True))
      elif 'fs' in incr_update:
        incr_update_json['storage'] = 'fs'
        incr_update_json['fs'] = {'incr_save_dir': incr_update['fs'].mount_path}
      json.dump(incr_update_json, fout, indent=2)

    ops.add_to_collection(
        ops.GraphKeys.ASSET_FILEPATHS,
        tf.constant(asset_file_path, dtype=tf.string, name=asset_file))

  export_dir = os.path.join(export_dir,
                            meta_graph_def.meta_info_def.meta_graph_version)
  export_dir = io_util.fix_oss_dir(export_dir)
  logging.info('export_dir=%s' % export_dir)
  if Exists(export_dir):
    logging.info('will delete old dir: %s' % export_dir)
    DeleteRecursively(export_dir)

  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  tensor_info_inputs = {}
  for tmp_key in inputs:
    tmp = graph.get_tensor_by_name(inputs[tmp_key].name)
    tensor_info_inputs[tmp_key] = \
        tf.saved_model.utils.build_tensor_info(tmp)

  tensor_info_outputs = {}
  for tmp_key in outputs:
    tmp = graph.get_tensor_by_name(outputs[tmp_key].name)
    tensor_info_outputs[tmp_key] = \
        tf.saved_model.utils.build_tensor_info(tmp)
  signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs=tensor_info_inputs,
          outputs=tensor_info_outputs,
          method_name=signature_constants.PREDICT_METHOD_NAME))

  if 'incr_update' in oss_params:
    incr_update_inputs = meta_graph_editor.sparse_update_inputs
    incr_update_outputs = meta_graph_editor.sparse_update_outputs
    incr_update_inputs.update(meta_graph_editor.dense_update_inputs)
    incr_update_outputs.update(meta_graph_editor.dense_update_outputs)
    tensor_info_incr_update_inputs = {}
    tensor_info_incr_update_outputs = {}
    for tmp_key in incr_update_inputs:
      tmp = graph.get_tensor_by_name(incr_update_inputs[tmp_key].name)
      tensor_info_incr_update_inputs[tmp_key] = \
          tf.saved_model.utils.build_tensor_info(tmp)
    for tmp_key in incr_update_outputs:
      tmp = graph.get_tensor_by_name(incr_update_outputs[tmp_key].name)
      tensor_info_incr_update_outputs[tmp_key] = \
          tf.saved_model.utils.build_tensor_info(tmp)
    incr_update_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=tensor_info_incr_update_inputs,
            outputs=tensor_info_incr_update_outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME))
  else:
    incr_update_signature = None

  session_config = ConfigProto(
      allow_soft_placement=True, log_device_placement=True)

  saver = tf.train.Saver()
  with tf.Session(target=server.target if server else '') as sess:
    saver.restore(sess, checkpoint_path)
    main_op = tf.group([
        Scaffold.default_local_init_op(),
        ops.get_collection(EMBEDDING_INITIALIZERS)
    ])
    incr_update_sig_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
    }
    if incr_update_signature is not None:
      incr_update_sig_map[INCR_UPDATE_SIGNATURE_KEY] = incr_update_signature
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map=incr_update_sig_map,
        assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        saver=saver,
        main_op=main_op,
        strip_default_attrs=True,
        clear_devices=True)
    builder.save()

  # remove temporary files
  Remove(embed_name_to_id_file)
  return export_dir
