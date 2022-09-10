# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import sys
import traceback

import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureGroupConfig
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep
from easy_rec.python.protos.pipeline_pb2 import EasyRecConfig
from easy_rec.python.utils import config_util

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

MAX_HASH_BUCKET_SIZE = 9223372036854775807


def _gen_raw_config(feature, input_field, feature_config, is_multi,
                    curr_embed_dim):
  if 'bucketize_boundaries' in feature:
    if is_multi:
      input_field.input_type = DatasetConfig.STRING
      feature_config.feature_type = feature_config.TagFeature
    else:
      input_field.input_type = DatasetConfig.INT32
      feature_config.feature_type = feature_config.IdFeature
    feature_config.num_buckets = len(
        feature['bucketize_boundaries'].split(',')) + 1
    feature_config.embedding_dim = curr_embed_dim
  else:
    feature_config.feature_type = feature_config.RawFeature
    input_field.default_val = str(feature.get('default_value', '0.0'))
    raw_input_dim = feature.get('value_dimension', 1)
    if raw_input_dim > 1:
      feature_config.raw_input_dim = raw_input_dim
      input_field.input_type = DatasetConfig.STRING
    else:
      input_field.input_type = DatasetConfig.DOUBLE
    if 'boundaries' in feature:
      feature_config.boundaries.extend(feature['boundaries'])
      feature_config.embedding_dim = curr_embed_dim
  if 'normalizer_fn' in feature:
    feature_config.normalizer_fn = feature['normalizer_fn']


def _set_hash_bucket(feature, feature_config, input_field):
  if 'max_partitions' in feature:
    feature_config.max_partitions = feature['max_partitions']
  if 'hash_bucket_size' in feature:
    feature_config.hash_bucket_size = feature['hash_bucket_size']
    if feature_config.hash_bucket_size > 10000000:
      if 'max_partitions' not in feature:
        logging.error(
            'it is suggested to set max_partitions > 1 for large hash buckets[%s]'
            % feature['feature_name'])
        sys.exit(1)
    if feature.get('filter_freq', -1) >= 0:
      feature_config.ev_params.filter_freq = feature['filter_freq']
      feature_config.hash_bucket_size = MAX_HASH_BUCKET_SIZE
    if feature.get('steps_to_live', -1) >= 0:
      feature_config.ev_params.steps_to_live = feature['steps_to_live']
      feature_config.hash_bucket_size = MAX_HASH_BUCKET_SIZE
  elif 'vocab_file' in feature:
    feature_config.vocab_file = feature['vocab_file']
  elif 'vocab_list' in feature:
    feature_config.vocab_list = feature['vocab_list']
  elif 'num_buckets' in feature:
    feature_config.num_buckets = feature['num_buckets']
    input_field.default_val = feature.get('default_value', '0')
  else:
    assert False, 'one of hash_bucket_size,vocab_file,vocab_list,num_buckets must be set'


def process_features(feature_type,
                     feature_name,
                     feature,
                     pipeline_config,
                     embedding_dim,
                     incol_separator,
                     is_sequence=False):
  feature_config = FeatureConfig()
  feature_config.input_names.append(feature_name)
  feature_config.separator = incol_separator
  input_field = DatasetConfig.Field()
  input_field.input_name = feature_name
  curr_embed_dim = feature.get('embedding_dimension',
                               feature.get('embedding_dim', embedding_dim))
  curr_combiner = feature.get('combiner', 'sum')
  if feature.get('is_cache', False):
    logging.info('will cache %s' % feature_name)
    feature_config.is_cache = True
  is_multi = feature.get('is_multi', False)
  # is_seq = feature.get('is_seq', False)
  if is_sequence:
    feature_config.feature_type = feature_config.SequenceFeature
    feature_config.embedding_dim = curr_embed_dim
    if feature_type == 'raw_feature':
      feature_config.sub_feature_type = feature_config.RawFeature
      input_field.default_val = feature.get('default_value', '0.0')
      raw_input_dim = feature.get('value_dimension', 1)
      if 'boundaries' in feature:
        feature_config.boundaries.extend(feature['boundaries'])
      if raw_input_dim > 1:
        feature_config.raw_input_dim = raw_input_dim
    else:
      feature_config.sub_feature_type = feature_config.IdFeature
      _set_hash_bucket(feature, feature_config, input_field)
      feature_config.combiner = curr_combiner
  elif feature_type == 'id_feature':
    if is_multi:
      feature_config.feature_type = feature_config.TagFeature
      kv_separator = feature.get('kv_separator', None)
      if kv_separator:
        feature_config.kv_separator = kv_separator
    # elif is_seq:
    #   feature_config.feature_type = feature_config.SequenceFeature
    else:
      feature_config.feature_type = feature_config.IdFeature
    feature_config.embedding_dim = curr_embed_dim
    _set_hash_bucket(feature, feature_config, input_field)
    feature_config.combiner = curr_combiner
  elif feature_type == 'lookup_feature':
    need_discrete = feature.get('needDiscrete', True)
    if not need_discrete:
      _gen_raw_config(feature, input_field, feature_config, is_multi,
                      curr_embed_dim)
    else:
      feature_config.feature_type = feature_config.TagFeature
      if feature.get('needWeighting', False):
        feature_config.kv_separator = ''
      feature_config.embedding_dim = curr_embed_dim
      _set_hash_bucket(feature, feature_config, input_field)
      feature_config.combiner = curr_combiner
  elif feature_type == 'raw_feature':
    _gen_raw_config(feature, input_field, feature_config, is_multi,
                    curr_embed_dim)
  elif feature_type == 'match_feature':
    need_discrete = feature.get('needDiscrete', True)
    if feature.get('matchType', '') == 'multihit':
      is_multi = True
    if need_discrete:
      feature_config.feature_type = feature_config.TagFeature
      if feature.get('needWeighting', False):
        feature_config.kv_separator = ''
      feature_config.embedding_dim = curr_embed_dim
      _set_hash_bucket(feature, feature_config, input_field)
      feature_config.combiner = curr_combiner
    else:
      assert 'bucketize_boundaries' not in feature
      _gen_raw_config(feature, input_field, feature_config, is_multi,
                      curr_embed_dim)
  elif feature_type == 'combo_feature':
    feature_config.feature_type = feature_config.TagFeature
    _set_hash_bucket(feature, feature_config, input_field)
    feature_config.embedding_dim = curr_embed_dim
    feature_config.combiner = curr_combiner
  elif feature_type == 'overlap_feature':
    if feature['method'] in ['common_word_divided', 'diff_word_divided']:
      feature_config.feature_type = feature_config.TagFeature
    else:
      feature_config.feature_type = feature_config.IdFeature
    _set_hash_bucket(feature, feature_config, input_field)
    feature_config.embedding_dim = curr_embed_dim
    feature_config.combiner = curr_combiner
  else:
    assert 'unknown feature type %s, currently not supported' % feature_type
  if 'shared_name' in feature:
    feature_config.embedding_name = feature['shared_name']
  # pipeline_config.feature_configs.append(feature_config)
  if pipeline_config.feature_configs:
    pipeline_config.feature_configs.append(feature_config)
  else:
    pipeline_config.feature_config.features.append(feature_config)
  pipeline_config.data_config.input_fields.append(input_field)

  if 'extra_combo_info' in feature:
    extra_combo_info = feature['extra_combo_info']
    feature_names = extra_combo_info.get('feature_names', [])
    assert len(
        feature_names
    ) >= 1, 'The feature number for ComboFeature must be greater than 2.'
    combo_feature_config = FeatureConfig()
    combo_feature_config.input_names.append(feature_name)

    for fea_name in feature_names:
      combo_feature_config.input_names.append(fea_name)

    final_feature_name = 'combo__' + '_'.join(combo_feature_config.input_names)
    final_feature_name = extra_combo_info.get('final_feature_name',
                                              final_feature_name)
    combo_feature_config.feature_name = final_feature_name
    combo_feature_config.feature_type = combo_feature_config.ComboFeature
    curr_embed_dim = extra_combo_info.get(
        'embedding_dimension',
        extra_combo_info.get('embedding_dim', embedding_dim))
    curr_combiner = extra_combo_info.get('combiner', 'mean')
    combo_feature_config.embedding_dim = curr_embed_dim
    combo_feature_config.combiner = curr_combiner
    assert 'hash_bucket_size' in extra_combo_info, 'hash_bucket_size must be set in ComboFeature.'
    _set_hash_bucket(extra_combo_info, combo_feature_config, None)

    if pipeline_config.feature_configs:
      pipeline_config.feature_configs.append(combo_feature_config)
    else:
      pipeline_config.feature_config.features.append(combo_feature_config)
  return pipeline_config


def load_input_field_and_feature_config(rtp_fg,
                                        label_fields,
                                        embedding_dim=16,
                                        incol_separator='\003'):
  embedding_dim = rtp_fg.get('embedding_dim', embedding_dim)
  logging.info('embedding_dim = %s' % embedding_dim)
  logging.info('label_fields = %s' % ','.join(label_fields))

  pipeline_config = EasyRecConfig()
  for tmp_lbl in label_fields:
    input_field = DatasetConfig.Field()
    input_field.input_name = tmp_lbl
    input_field.input_type = DatasetConfig.INT32
    input_field.default_val = '0'
    pipeline_config.data_config.input_fields.append(input_field)

  rtp_features = rtp_fg['features']
  for feature in rtp_features:
    logging.info('feature type = %s' % type(feature))
    logging.info('feature = %s' % feature)
    logging.info('feature_type in feature %s' % ('feature_name' in feature))
    try:
      if 'feature_name' in feature:
        feature_type = feature['feature_type']
        feature_name = feature['feature_name']
        pipeline_config = process_features(feature_type, feature_name, feature,
                                           pipeline_config, embedding_dim,
                                           incol_separator)
      elif 'sequence_name' in feature:
        logging.info('Set sequence_features group later.')
        sequence_name = feature['sequence_name']
        for sub_feature in feature['features']:
          sub_feature_type = sub_feature['feature_type']
          sub_feature_name = sub_feature['feature_name']
          all_sub_feature_name = sequence_name + '_' + sub_feature_name
          pipeline_config = process_features(
              sub_feature_type,
              all_sub_feature_name,
              sub_feature,
              pipeline_config,
              embedding_dim,
              incol_separator,
              is_sequence=True)
    except Exception:
      logging.info('convert feature[%s] exception[%s]' %
                   (str(feature), traceback.format_exc()))
      sys.exit(1)
  return pipeline_config


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
                   selected_cols='',
                   input_type='OdpsRTPInput',
                   is_async=False):
  with tf.gfile.GFile(rtp_fg, 'r') as fin:
    rtp_fg = json.load(fin)

  model_dir = rtp_fg.get('model_dir', 'experiments/rtp_fg_demo')
  num_steps = rtp_fg.get('num_steps', num_steps)
  model_type = rtp_fg.get('model_type', model_type)
  label_fields = rtp_fg.get('label_fields', label_fields)
  model_path = rtp_fg.get('model_path', '')
  edit_config_json = rtp_fg.get('edit_config_json', None)
  rtp_features = rtp_fg['features']

  logging.info('model_dir = %s' % model_dir)
  logging.info('num_steps = %d' % num_steps)
  logging.info('model_type = %s' % model_type)
  logging.info('model_path = %s' % model_path)
  logging.info('edit_config_json = %s' % edit_config_json)

  pipeline_config = load_input_field_and_feature_config(rtp_fg, label_fields,
                                                        embedding_dim,
                                                        incol_separator)
  pipeline_config.model_dir = model_dir
  pipeline_config.data_config.separator = separator
  if selected_cols:
    pipeline_config.data_config.selected_cols = selected_cols
  if train_input_path is not None:
    pipeline_config.train_input_path = train_input_path
  if eval_input_path is not None:
    pipeline_config.eval_input_path = eval_input_path

  pipeline_config.data_config.batch_size = batch_size
  pipeline_config.data_config.rtp_separator = ';'
  pipeline_config.data_config.label_fields.extend(label_fields)

  text_format.Merge('input_type: %s' % input_type, pipeline_config.data_config)

  if model_path:
    model_type = None
    with tf.gfile.GFile(model_path, 'r') as fin:
      model_config = fin.read()
      text_format.Merge(model_config, pipeline_config)

  if not pipeline_config.HasField('train_config'):
    train_config_str = """
    train_config {
      log_step_count_steps: 200
      optimizer_config: {
        %s: {
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

      sync_replicas: %s
    }
    """ % ('adam_optimizer' if not is_async else 'adam_async_optimizer',
           'true' if not is_async else 'false')
    text_format.Merge(train_config_str, pipeline_config)

  pipeline_config.train_config.num_steps = num_steps

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
  elif model_type == 'wide_and_deep':
    pipeline_config.model_config.model_class = 'WideAndDeep'
    wide_group = FeatureGroupConfig()
    wide_group.group_name = 'wide'
    wide_group.wide_deep = WideOrDeep.WIDE
    for feature in rtp_features:
      feature_name = feature['feature_name']
      group = feature.get('group', 'wide_and_deep')
      if group not in ['wide', 'deep', 'wide_and_deep']:
        logging.warning('invalid group %s for %s' % (group, feature_name))
        group = 'wide_and_deep'
      if group in ['wide', 'wide_and_deep']:
        wide_group.feature_names.append(feature_name)
    pipeline_config.model_config.feature_groups.append(wide_group)
    deep_group = FeatureGroupConfig()
    deep_group.group_name = 'deep'
    deep_group.wide_deep = WideOrDeep.DEEP
    for feature in rtp_features:
      feature_name = feature['feature_name']
      group = feature.get('group', 'wide_and_deep')
      if group not in ['wide', 'deep', 'wide_and_deep']:
        group = 'wide_and_deep'
      if group in ['deep', 'wide_and_deep']:
        deep_group.feature_names.append(feature_name)
    pipeline_config.model_config.feature_groups.append(deep_group)
    deepfm_config_str = """
    wide_and_deep {
      dnn {
        hidden_units: [128, 64, 32]
      }
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

  elif model_type == 'esmm':
    pipeline_config.model_config.model_class = 'ESMM'

    feature_groups = {}
    for feature in rtp_features:
      feature_name = feature['feature_name']
      group = feature.get('group', 'all')
      if group in feature_groups:
        feature_groups[group].append(feature_name)
      else:
        feature_groups[group] = [feature_name]

    for group_name in feature_groups:
      logging.info('add group = %s' % group_name)
      group = FeatureGroupConfig()
      group.group_name = group_name
      for fea_name in feature_groups[group_name]:
        group.feature_names.append(fea_name)
      group.wide_deep = WideOrDeep.DEEP
      pipeline_config.model_config.feature_groups.append(group)

    esmm_config_str = '  esmm {\n'
    for group_name in feature_groups:
      esmm_config_str += """
        groups {
          input: "%s"
          dnn {
            hidden_units: [256, 128, 96, 64]
          }
        }""" % group_name

    esmm_config_str += """
        ctr_tower {
          tower_name: "ctr"
          label_name: "%s"
          dnn {
            hidden_units: [128, 96, 64, 32, 16]
          }
          num_class: 1
          weight: 1.0
          loss_type: CLASSIFICATION
          metrics_set: {
           auc {}
          }
        }
        cvr_tower {
          tower_name: "cvr"
          label_name: "%s"
          dnn {
            hidden_units: [128, 96, 64, 32, 16]
          }
          num_class: 1
          weight: 1.0
          loss_type: CLASSIFICATION
          metrics_set: {
           auc {}
          }
        }
        l2_regularization: 1e-6
      }""" % (label_fields[0], label_fields[1])
    text_format.Merge(esmm_config_str, pipeline_config.model_config)
    pipeline_config.model_config.embedding_regularization = 5e-5
  elif model_type == 'dbmtl':
    pipeline_config.model_config.model_class = 'DBMTL'

    feature_groups = {}
    for feature in rtp_features:
      feature_name = feature['feature_name']
      group = 'all'
      if group in feature_groups:
        feature_groups[group].append(feature_name)
      else:
        feature_groups[group] = [feature_name]

    for group_name in feature_groups:
      logging.info('add group = %s' % group_name)
      group = FeatureGroupConfig()
      group.group_name = group_name
      for fea_name in feature_groups[group_name]:
        group.feature_names.append(fea_name)
      group.wide_deep = WideOrDeep.DEEP
      pipeline_config.model_config.feature_groups.append(group)

    dbmtl_config_str = """
      dbmtl {
        bottom_dnn {
          hidden_units: [1024]
        }
        expert_dnn {
          hidden_units: [256, 128, 64, 32]
        }
        num_expert: 8
        task_towers {
          tower_name: "ctr"
          label_name: "%s"
          loss_type: CLASSIFICATION
          metrics_set: {
            auc {}
          }
          dnn {
            hidden_units: [256, 128, 64, 32]
          }
          relation_dnn {
            hidden_units: [32]
          }
          weight: 1.0
        }
        task_towers {
          tower_name: "cvr"
          label_name: "%s"
          loss_type: CLASSIFICATION
          metrics_set: {
            auc {}
          }
          dnn {
            hidden_units: [256, 128, 64, 32]
          }
          relation_tower_names: ["ctr"]
          relation_dnn {
            hidden_units: [32]
          }
          weight: 1.0
        }
        l2_regularization: 1e-6
      }
    """ % (label_fields[0], label_fields[1])
    text_format.Merge(dbmtl_config_str, pipeline_config.model_config)
    pipeline_config.model_config.embedding_regularization = 5e-6

  if model_type in ['wide_and_deep', 'deepfm', 'multi_tower']:
    text_format.Merge("""
      metrics_set {
        auc {}
      }
      """, pipeline_config.eval_config)

  text_format.Merge(
      """ export_config {
          multi_placeholder: false
        }
    """, pipeline_config)

  if edit_config_json:
    for edit_obj in edit_config_json:
      config_util.edit_config(pipeline_config, edit_obj)

    pipeline_config.model_config.embedding_regularization = 1e-5
  return pipeline_config
