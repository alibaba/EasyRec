# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Functions for reading and updating configuration files.

Such as Hyper parameter tuning or automatic feature expanding.
"""

import json
import logging
import os
import re
import sys

import six
import tensorflow as tf
from google.protobuf import json_format
from google.protobuf import text_format
from tensorflow.python.lib.io import file_io

from easy_rec.python.protos import pipeline_pb2
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureGroupConfig
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep
from easy_rec.python.protos.pipeline_pb2 import EasyRecConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def get_configs_from_pipeline_file(pipeline_config_path, auto_expand=True):
  """Reads config from a file containing pipeline_pb2.EasyRecConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.EasyRecConfig text
      proto.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  if isinstance(pipeline_config_path, pipeline_pb2.EasyRecConfig):
    return pipeline_config_path

  assert tf.gfile.Exists(
      pipeline_config_path
  ), 'pipeline_config_path [%s] not exists' % pipeline_config_path

  pipeline_config = pipeline_pb2.EasyRecConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    config_str = f.read()
    if pipeline_config_path.endswith('.config'):
      text_format.Merge(config_str, pipeline_config)
    elif pipeline_config_path.endswith('.json'):
      json_format.Parse(config_str, pipeline_config)
    else:
      assert 'invalid file format(%s), currently support formats: .config(prototxt) .json' % pipeline_config_path

  if auto_expand:
    return auto_expand_share_feature_configs(pipeline_config)
  else:
    return pipeline_config


def auto_expand_share_feature_configs(pipeline_config):
  for share_config in pipeline_config.feature_configs:
    if len(share_config.shared_names) == 0:
      continue

    # auto expand all shared_names
    input_names = []
    for input_name in share_config.shared_names:
      if pipeline_config.data_config.auto_expand_input_fields:
        input_names.extend(auto_expand_names(input_name))
      else:
        input_names.append(input_name)

    # make a clean copy
    while len(share_config.shared_names) > 0:
      share_config.shared_names.pop()

    fea_config = FeatureConfig()
    fea_config.CopyFrom(share_config)
    while len(fea_config.input_names) > 0:
      fea_config.input_names.pop()

    # generate for each item in input_name
    for tmp_name in input_names:
      tmp_config = FeatureConfig()
      tmp_config.CopyFrom(fea_config)
      tmp_config.input_names.append(tmp_name)
      pipeline_config.feature_configs.append(tmp_config)
  return pipeline_config


def auto_expand_names(input_name):
  """Auto expand field[1-3] to field1, field2, field3.

  Args:
    input_name: a string pattern like field[1-3]

  Returns:
    a string list of the expanded names
  Todo:
    could be extended to support more complicated patterns
  """
  match_obj = re.match(r'([a-zA-Z_]+)\[([0-9]+)-([0-9]+)\]', input_name)
  if match_obj:
    prefix = match_obj.group(1)
    sid = int(match_obj.group(2))
    eid = int(match_obj.group(3)) + 1
    input_name = ['%s%d' % (prefix, tid) for tid in range(sid, eid)]
  else:
    input_name = [input_name]
  return input_name


def create_pipeline_proto_from_configs(configs):
  """Creates a pipeline_pb2.EasyRecConfig from configs dictionary.

  This function performs the inverse operation of
  create_configs_from_pipeline_proto().

  Args:
    configs: Dictionary of configs. See get_configs_from_pipeline_file().

  Returns:
    A fully populated pipeline_pb2.EasyRecConfig.
  """
  pipeline_config = pipeline_pb2.EasyRecConfig()
  pipeline_config.model.CopyFrom(configs['model'])
  pipeline_config.train_config.CopyFrom(configs['train_config'])
  pipeline_config.train_input_reader.CopyFrom(configs['train_input_config'])
  pipeline_config.eval_config.CopyFrom(configs['eval_config'])
  pipeline_config.eval_input_reader.CopyFrom(configs['eval_input_config'])
  if 'graph_rewriter_config' in configs:
    pipeline_config.graph_rewriter.CopyFrom(configs['graph_rewriter_config'])
  return pipeline_config


def save_pipeline_config(pipeline_config,
                         directory,
                         filename='pipeline.config'):
  """Saves a pipeline config text file to disk.

  Args:
    pipeline_config: A pipeline_pb2.TrainEvalPipelineConfig.
    directory: The model directory into which the pipeline config file will be
      saved.
    filename: pipelineconfig filename
  """
  if not file_io.file_exists(directory):
    file_io.recursive_create_dir(directory)
  pipeline_config_path = os.path.join(directory, filename)
  # as_utf8=True to make sure pbtxt is human readable when string contains chinese
  save_message(pipeline_config, pipeline_config_path)


def edit_config(pipeline_config, edit_config_json):
  """Update params specified by automl.

  Args:
    pipeline_config: EasyRecConfig
    edit_config_json: edit config json
  """

  def _get_attr(obj, attr, only_last=False):
    # only_last means we only return the last element in paths array
    attr_toks = [x.strip() for x in attr.split('.') if x != '']
    paths = []
    objs = [obj]
    nobjs = []
    for key in attr_toks:
      # clear old paths to clear new paths
      paths = []
      for obj in objs:
        if '[' in key:
          pos = key.find('[')
          name, cond = key[:pos], key[pos + 1:]
          cond = cond[:-1]
          update_objs = getattr(obj, name)
          # select all update_objs
          if cond == ':':
            for tid, update_obj in enumerate(update_objs):
              paths.append((obj, update_obj, None, tid))
              nobjs.append(update_obj)
            continue

          # select by range update_objs[1:10]
          if ':' in cond:
            colon_pos = cond.find(':')
            sid = cond[:colon_pos]
            if len(sid) == 0:
              sid = 0
            else:
              sid = int(sid)
            eid = cond[(colon_pos + 1):]
            if len(eid) == 0:
              eid = len(update_objs)
            else:
              eid = int(eid)
            for tid, update_obj in enumerate(update_objs[sid:eid]):
              paths.append((obj, update_obj, None, tid + sid))
              nobjs.append(update_obj)
            continue

          # for simple index update_objs[0]
          try:
            obj_id = int(cond)
            obj = update_objs[obj_id]
            paths.append((obj, update_objs, None, obj_id))
            continue
          except ValueError:
            pass

          # for complex conditions a[optimizer.lr=20]
          op_func_map = {
              '>=': lambda x, y: x >= y,
              '<=': lambda x, y: x <= y,
              '<': lambda x, y: x < y,
              '>': lambda x, y: x > y,
              '=': lambda x, y: x == y
          }
          cond_key = None
          cond_val = None
          op_func = None
          for op in ['>=', '<=', '>', '<', '=']:
            tmp_pos = cond.rfind(op)
            if tmp_pos != -1:
              cond_key = cond[:tmp_pos]
              cond_val = cond[(tmp_pos + len(op)):]
              op_func = op_func_map[op]
              break

          assert cond_key is not None, 'invalid cond: %s' % cond
          assert cond_val is not None, 'invalid cond: %s' % cond

          for tid, update_obj in enumerate(update_objs):
            tmp, _, _, _ = _get_attr(update_obj, cond_key, only_last=True)
            if type(cond_val) != type(tmp):
              cond_val = type(tmp)(cond_val)
            if op_func(tmp, cond_val):
              obj_id = tid
              paths.append((update_obj, update_objs, None, obj_id))
              nobjs.append(update_obj)
        else:
          sub_obj = getattr(obj, key)
          paths.append((sub_obj, obj, key, -1))
          nobjs.append(sub_obj)
      # exchange to prepare for parsing next token
      objs = nobjs
      nobjs = []
    if only_last:
      return paths[-1]
    else:
      return paths

  for param_keys in edit_config_json:
    # multiple keys/vals combination
    param_vals = edit_config_json[param_keys]
    param_vals = [x.strip() for x in str(param_vals).split(';')]
    param_keys = [x.strip() for x in str(param_keys).split(';')]
    for param_key, param_val in zip(param_keys, param_vals):
      update_obj = pipeline_config
      tmp_paths = _get_attr(update_obj, param_key)
      # update a set of objs
      for tmp_val, tmp_obj, tmp_name, tmp_id in tmp_paths:
        basic_types = [int, str, float, bool]
        if six.PY2:
          basic_types.append(unicode)  # noqa: F821
        if type(tmp_val) in basic_types:
          # simple type cast
          tmp_val = type(tmp_val)(param_val)
          if tmp_name is None:
            tmp_obj[tmp_id] = tmp_val
          else:
            setattr(tmp_obj, tmp_name, tmp_val)
        elif 'RepeatedScalarContainer' in str(type(tmp_val)):
          tmp_obj.ClearField(tmp_name)
          text_format.Parse('%s:%s' % (tmp_name, param_val), tmp_obj)
        else:
          tmp_val.Clear()
          param_val = param_val.strip()
          if param_val.startswith('{') and param_val.endswith('}'):
            param_val = param_val[1:-1]
          text_format.Parse(param_val, tmp_val)

  return pipeline_config


def save_message(protobuf_message, filename):
  """Saves a pipeline config text file to disk.

  Args:
    protobuf_message: A pipeline_pb2.TrainEvalPipelineConfig.
    filename: pipeline config filename
  """
  directory, _ = os.path.split(filename)
  if not file_io.file_exists(directory):
    file_io.recursive_create_dir(directory)
  # as_utf8=True to make sure pbtxt is human readable when string contains chinese
  config_text = text_format.MessageToString(protobuf_message, as_utf8=True)
  with tf.gfile.Open(filename, 'wb') as f:
    logging.info('Writing protobuf message file to %s', filename)
    f.write(config_text)


def convert_rtp_fg(rtp_fg,
                   embedding_dim=16,
                   batch_size=1024,
                   label_fields=[],
                   num_steps=10,
                   model_type='',
                   separator='\002',
                   incol_separator='\003',
                   train_input_path=None,
                   eval_input_path=None,
                   selected_cols=''):
  pipeline_config = EasyRecConfig()
  with tf.gfile.GFile(rtp_fg, 'r') as fin:
    rtp_fg = json.load(fin)
  for tmp_lbl in label_fields:
    input_field = DatasetConfig.Field()
    input_field.input_name = tmp_lbl
    input_field.input_type = DatasetConfig.INT32
    input_field.default_val = '0'
    pipeline_config.data_config.input_fields.append(input_field)

  pipeline_config.data_config.separator = separator
  if selected_cols:
    pipeline_config.data_config.selected_cols = selected_cols
  if train_input_path is not None:
    pipeline_config.train_input_path = train_input_path
  if eval_input_path is not None:
    pipeline_config.eval_input_path = eval_input_path
  pipeline_config.model_dir = 'experiments/rtp_fg_demo'

  rtp_features = rtp_fg['features']
  for feature in rtp_features:
    try:
      feature_type = feature['feature_type']
      feature_name = feature['feature_name']
      feature_config = FeatureConfig()
      feature_config.input_names.append(feature_name)
      feature_config.separator = incol_separator
      input_field = DatasetConfig.Field()
      input_field.input_name = feature_name
      if feature_type == 'id_feature':
        feature_config.feature_type = feature_config.TagFeature
        feature_config.embedding_dim = embedding_dim
        feature_config.hash_bucket_size = feature['hash_bucket_size']
      elif feature_type == 'lookup_feature':
        need_discrete = feature.get('needDiscrete', True)
        need_key = feature.get('needKey', True)  # noqa: F841
        if not need_discrete:
          feature_config.feature_type = feature_config.RawFeature
          input_field.input_type = DatasetConfig.DOUBLE
          input_field.default_val = '0.0'
        else:
          feature_config.feature_type = feature_config.TagFeature
          feature_config.embedding_dim = embedding_dim
          feature_config.hash_bucket_size = feature['hash_bucket_size']
      elif feature_type == 'raw_feature':
        feature_config.feature_type = feature_config.RawFeature
        input_field.input_type = DatasetConfig.DOUBLE
        input_field.default_val = '0.0'
      elif feature_type == 'match_feature':
        feature_config.input_names.append(feature_name + '_wgt')
        feature_config.feature_type = feature_config.TagFeature
        feature_config.embedding_dim = embedding_dim
        feature_config.hash_bucket_size = feature['hash_bucket_size']
      elif feature_type == 'combo_feature':
        feature_config.feature_type = feature_config.TagFeature
        feature_config.hash_bucket_size = feature['hash_bucket_size']
        feature_config.embedding_dim = embedding_dim
      elif feature_type == 'overlap_feature':
        if feature['method'] in ['common_word_divided', 'diff_word_divided']:
          feature_config.feature_type = feature_config.TagFeature
        else:
          feature_config.feature_type = feature_config.IdFeature
        feature_config.hash_bucket_size = feature['hash_bucket_size']
        feature_config.embedding_dim = embedding_dim
      elif feature_type == 'expr_feature':
        feature_config.feature_type = feature_config.RawFeature
        input_field.input_type = DatasetConfig.DOUBLE
        input_field.default_val = '0.0'
      else:
        assert 'unknown feature type %s, currently not supported' % feature_type
      if 'shared_name' in feature:
        feature_config.embedding_name = feature['shared_name']
      pipeline_config.feature_configs.append(feature_config)
      pipeline_config.data_config.input_fields.append(input_field)
    except Exception as ex:
      print('Exception: %s %s' % (type(ex), str(ex)))
      print(feature)
      sys.exit(1)
  pipeline_config.data_config.input_type = pipeline_config.data_config.RTPInput
  pipeline_config.data_config.batch_size = batch_size
  pipeline_config.data_config.rtp_separator = ';'
  pipeline_config.data_config.label_fields.extend(label_fields)
  pipeline_config.train_config.num_steps = num_steps

  if model_type:
    train_config_str = """
    train_config {
      log_step_count_steps: 200
      optimizer_config: {
        adam_optimizer: {
          learning_rate: {
            exponential_decay_learning_rate {
              initial_learning_rate: 0.0001
              decay_steps: 100000
              decay_factor: 0.5
              min_learning_rate: 0.0000001
            }
          }
        }
        use_moving_average: false
      }

      sync_replicas: true
    }
    """
    text_format.Merge(train_config_str, pipeline_config)

  if model_type == 'deepfm':
    pipeline_config.model_config.model_class = 'DeepFM'
    wide_group = FeatureGroupConfig()
    wide_group.group_name = 'wide'
    wide_group.wide_deep = WideOrDeep.WIDE
    for feature in rtp_features:
      feature_name = feature['feature_name']
      wide_group.feature_names.append(feature_name)
    pipeline_config.model_config.feature_groups.append(wide_group)
    deep_group = FeatureGroupConfig()
    deep_group.CopyFrom(wide_group)
    deep_group.group_name = 'deep'
    deep_group.wide_deep = WideOrDeep.DEEP
    pipeline_config.model_config.feature_groups.append(deep_group)
    deepfm_config_str = """
    deepfm {
      dnn {
        hidden_units: [128, 64, 32]
      }
      final_dnn {
        hidden_units: [128, 64]
      }
      wide_output_dim: 32
      l2_regularization: 1e-5
    }
    """
    text_format.Merge(deepfm_config_str, pipeline_config.model_config)
    pipeline_config.model_config.embedding_regularization = 1e-5
  elif model_type == 'multi_tower':
    pipeline_config.model_config.model_class = 'MultiTower'

    feature_groups = {}
    group_map = {
        'u': 'user',
        'i': 'item',
        'ctx': 'combo',
        'q': 'combo',
        'comb': 'combo'
    }
    for feature in rtp_features:
      feature_name = feature['feature_name'].strip()
      group_name = ''
      if 'group' in feature:
        group_name = feature['group']
      else:
        toks = feature_name.split('_')
        group_name = toks[0]
        if group_name in group_map:
          group_name = group_map[group_name]
      if group_name in feature_groups:
        feature_groups[group_name].append(feature_name)
      else:
        feature_groups[group_name] = [feature_name]

    logging.info(
        'if group is specified, group will be used as feature group name; '
        'otherwise, the prefix of feature_name in fg.json is used as feature group name'
    )
    logging.info('prefix map: %s' % str(group_map))
    for group_name in feature_groups:
      logging.info('add group = %s' % group_name)
      group = FeatureGroupConfig()
      group.group_name = group_name
      for fea_name in feature_groups[group_name]:
        group.feature_names.append(fea_name)
      group.wide_deep = WideOrDeep.DEEP
      pipeline_config.model_config.feature_groups.append(group)

    multi_tower_config_str = '  multi_tower {\n'
    for group_name in feature_groups:
      multi_tower_config_str += """
      towers {
        input: "%s"
        dnn {
          hidden_units: [256, 192, 128]
        }
      }
      """ % group_name

    multi_tower_config_str = multi_tower_config_str + """
      final_dnn {
        hidden_units: [192, 128, 64]
      }
      l2_regularization: 1e-4
    }
    """
    text_format.Merge(multi_tower_config_str, pipeline_config.model_config)
    pipeline_config.model_config.embedding_regularization = 1e-5
    text_format.Merge("""
    metrics_set {
      auc {}
    }
    """, pipeline_config.eval_config)

    text_format.Merge(
        """
      export_config {
        multi_placeholder: false
      }
    """, pipeline_config)
  return pipeline_config
