import json
import logging

import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.utils.config_util import get_compatible_feature_configs

from easy_rec.python.utils.convert_rtp_fg import load_input_field_and_feature_config  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def load_fg_json_to_config(pipeline_config):
  fg_json_path = pipeline_config.fg_json_path
  if not fg_json_path:
    return

  if fg_json_path.startswith('!'):
    # already loaded
    return

  label_fields = pipeline_config.data_config.label_fields
  with tf.gfile.GFile(fg_json_path, 'r') as fin:
    rtp_fg = json.load(fin)

  fg_config = load_input_field_and_feature_config(
      rtp_fg, label_fields=label_fields)

  pipeline_config.data_config.ClearField('input_fields')
  pipeline_config.ClearField('feature_configs')

  # not clear features so that we could define extra features
  # which is not defined in fg.json
  # pipeline_config.feature_config.ClearField('features')

  for input_config in fg_config.data_config.input_fields:
    in_config = DatasetConfig.Field()
    in_config.CopyFrom(input_config)
    pipeline_config.data_config.input_fields.append(in_config)
  fg_fea_config = get_compatible_feature_configs(fg_config)
  for fc in fg_fea_config:
    fea_config = FeatureConfig()
    fea_config.CopyFrom(fc)
    pipeline_config.feature_config.features.append(fea_config)
  logging.info('data_config and feature_config has been replaced by fg_json.')

  # signal that it is already loaded
  pipeline_config.fg_json_path = '!' + pipeline_config.fg_json_path

  return pipeline_config


def _fg(_fg_config,
        _effective_fg_features,
        _fg_module,
        field_dict,
        parsed_dict={}):
  multi_val_sep = _fg_config.get('multi_val_sep', '\035')
  input_dict = {}
  output_dict = {}

  def _tf_type(in_type):
    in_type = in_type.lower()
    type_map = {
        'integer': tf.int32,
        'int32': tf.int32,
        'int64': tf.int32,
        'bigint': tf.int64,
        'string': tf.string,
        'float': tf.float32,
        'double': tf.double
    }
    assert in_type in type_map, 'invalid type: %s' % in_type
    return type_map[in_type]

  def _get_input(input_name):
    if input_name in input_dict:
      return input_dict[input_name]

    sample_type = parsed_dict.get('__sampler_type__', None)

    side, key = input_name.split(':')
    x = field_dict[key]
    if sample_type is not None:
      num_neg = parsed_dict.get('__num_neg_sample__', None)
      batch_size = parsed_dict.get('__batch_size__', None)

      if sample_type.startswith('hard_negative_sampler'):
        raise NotImplementedError
      else:
        if side == 'user':
          x = tf.reshape(
              tf.tile(x[:, tf.newaxis], multiples=[1, 1 + num_neg]), [-1])
        elif side == 'item':
          x = tf.reshape(
              tf.concat([
                  x[:batch_size, tf.newaxis],
                  tf.tile(
                      x[tf.newaxis, batch_size:], multiples=[batch_size, 1])
              ],
                        axis=-1), [-1])  # noqa
        else:
          raise ValueError('Unknown side: %s' % side)
    input_dict[input_name] = x if x.dtype == tf.string else tf.as_string(x)
    return input_dict[input_name]

  for feature_config in _fg_config['features']:
    if 'sequence_name' in feature_config:
      sequence_name = feature_config['sequence_name']  # tag_category_list
      sequence_delim = feature_config.get('sequence_delim', ';')  # ";"
      for sub_feature_config in feature_config['features']:
        sub_feature_type = sub_feature_config['feature_type']  # id_feature
        sub_feature_name = sub_feature_config['feature_name']  # cate_id
        feature_name = sequence_name + '__' + sub_feature_name  # tag_category_list__cate_id
        if feature_name not in _effective_fg_features:
          continue
        if sub_feature_type == 'id_feature':
          # input = sequence_name + '__' + field_dict[sub_feature_config['expression'].split(':')[-1]]
          input = field_dict[feature_name]
          sparse_input = tf.string_split(
              input, delimiter=feature_config['sequence_delim'])
          seq_indices = tf.segment_max(
              tf.add(sparse_input.indices[:, 1], 1),
              sparse_input.indices[:, 0],
              name=None)
          batch_size = tf.shape(input)[0]
          pad_size = batch_size - tf.shape(seq_indices)[0]
          seq_indices_pad = tf.pad(seq_indices, [[0, pad_size]])
          sparse_input_values = sparse_input.values
          x = _fg_module.batch_sequence_id_feature_op(
              sparse_input_values,
              seq_indices_pad,
              feature_name=feature_name,
              msep=multi_val_sep,
              default_value=feature_config.get('default_value', ''),
              need_prefix=feature_config.get('need_prefix', False),
              sequence_delim=sequence_delim,
              dtype=tf.string)
          output_dict[feature_name] = x
          if parsed_dict.get('__sampler_type__', None) is not None:
            num_neg = parsed_dict.get('__num_neg_sample__', None)
            output_dict[feature_name] = tf.reshape(
                tf.tile(x[:, tf.newaxis], multiples=[1, 1 + num_neg]), [-1])
        elif sub_feature_type == 'raw_feature':
          # input = sequence_name + '__' + field_dict[sub_feature_config['expression'].split(':')[-1]]
          input = field_dict[feature_name]
          sparse_input = tf.string_split(
              input, delimiter=feature_config['sequence_delim'])
          seq_indices = tf.segment_max(
              tf.add(sparse_input.indices[:, 1], 1),
              sparse_input.indices[:, 0],
              name=None)
          batch_size = tf.shape(input)[0]
          pad_size = batch_size - tf.shape(seq_indices)[0]
          seq_indices_pad = tf.pad(seq_indices, [[0, pad_size]])
          sparse_input_values = sparse_input.values
          output_dict[feature_name] = _fg_module.batch_sequence_raw_feature_op(
              sparse_input_values,
              seq_indices_pad,
              feature_name=feature_name,
              msep=multi_val_sep,
              default_value=feature_config.get('default_value', '0.0'),
              sequence_delim=sequence_delim,
              normalizer=feature_config.get('normalizer', ''),
              value_dimension=feature_config.get('value_dimension', 1),
              dtype=tf.string)
        else:
          raise ValueError('Unknown seq sub feature type: %s' %
                           sub_feature_type)
    else:
      feature_type = feature_config['feature_type']
      feature_name = feature_config['feature_name']
      if feature_name not in _effective_fg_features:
        continue
      if feature_type == 'id_feature':
        output_dict[feature_name] = _fg_module.id_feature_op(
            _get_input(feature_config['expression']),
            feature_name=feature_name,
            msep=multi_val_sep,
            default_value=feature_config.get('default_value', '0.0'),
            need_prefix=feature_config.get('need_prefix', True),
            dtype=tf.string)
      elif feature_type == 'raw_feature':
        output_dict[feature_name] = _fg_module.raw_feature_op(
            _get_input(feature_config['expression']),
            feature_name=feature_name,
            msep=multi_val_sep,
            default_value=feature_config.get('default_value', '0.0'),
            normalizer=feature_config.get('normalizer', ''),
            value_dimension=feature_config.get('value_dimension', 1),
            dtype=_tf_type(feature_config.get('value_type', 'float')))
      elif feature_type == 'combo_feature':
        inputs = [_get_input(k) for k in feature_config['expression']]
        output_dict[feature_name] = _fg_module.combo_feature_op(
            inputs,
            feature_name=feature_name,
            msep=multi_val_sep,
            default_value=feature_config.get('default_value', ''),
            need_prefix=feature_config.get('need_prefix', True),
            dtype='string')
      elif feature_type == 'lookup_feature':
        output_dict[feature_name] = _fg_module.lookup_feature_op(
            _get_input(feature_config['map']),
            _get_input(feature_config['key']),
            feature_name=feature_name,
            msep=multi_val_sep,
            default_value=feature_config.get('default_value', '0.0'),
            dtype=_tf_type(feature_config.get('value_type', 'float')),
            need_discrete=feature_config.get('needDiscrete', False),
            need_key=feature_config.get('needKey', False),
            need_weighting=feature_config.get('needWeighting', False),
            value_dimension=feature_config.get('value_dimension', 1),
            combiner=feature_config.get('combiner', 'sum'),
            boundaries=feature_config.get('bucketize_boundaries', []),
            normalizer=feature_config.get('normalizer', ''))
      elif feature_type == 'match_feature':
        output_dict[feature_name] = _fg_module.match_feature_op(
            _get_input(feature_config['user']),
            _get_input(feature_config['category']),
            _get_input(feature_config['item']),
            feature_name=feature_name,
            msep=multi_val_sep,
            default_value=feature_config.get('default_value', '0.0'),
            dtype=_tf_type(feature_config.get('value_type', 'float')),
            need_discrete=feature_config.get('needDiscrete', False),
            normalizer=feature_config.get('normalizer', ''),
            match_type=feature_config.get('matchType', 'hit'))
      else:
        raise ValueError('Unknown feature type: %s' % feature_type)

  output_dict = dict(field_dict, **output_dict)
  return output_dict
