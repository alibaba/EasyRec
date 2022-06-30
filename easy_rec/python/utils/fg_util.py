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
