# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os
import re
import common_io
import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.utils import config_util

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)
tf.app.flags.DEFINE_string('template_config_path', None,
                           'Path to template pipeline config '
                           'file.')
tf.app.flags.DEFINE_string('output_config_path', None,
                           'Path to output pipeline config '
                           'file.')
tf.app.flags.DEFINE_string('config_table', '', 'config table')

FLAGS = tf.app.flags.FLAGS


def main(argv):
  pipeline_config = config_util.get_configs_from_pipeline_file(
      FLAGS.template_config_path)

  reader = common_io.table.TableReader(
      FLAGS.config_table, selected_cols='feature,feature_info')
  feature_info_map = {}
  while True:
    try:
      record = reader.read()
      feature_name = record[0][0]
      feature_info_map[feature_name] = json.loads(record[0][1])
    except common_io.exception.OutOfRangeException:
      reader.close()
      break

  for feature_config in config_util.get_compatible_feature_configs(
      pipeline_config):
    feature_name = feature_config.input_names[0]
    if feature_name in feature_info_map:
      logging.info('edited %s' % feature_name)
      feature_config.embedding_dim = int(
          feature_info_map[feature_name]['embedding_dim'])
      logging.info('embedding_dim: %s' % feature_config.embedding_dim)
      if 'boundary' in feature_info_map[feature_name]:
        feature_config.ClearField('boundaries')
        feature_config.boundaries.extend(
            [float(i) for i in feature_info_map[feature_name]['boundary']])
        logging.info('boundaries: %s' % feature_config.boundaries)
      elif 'hash_bucket_size' in feature_info_map[feature_name]:
        feature_config.hash_bucket_size = int(
            feature_info_map[feature_name]['hash_bucket_size'])
        logging.info('hash_bucket_size: %s' % feature_config.hash_bucket_size)

  pipeline_config.train_config.num_steps = feature_info_map['num_steps'][
      'num_steps']
  train_config = pipeline_config.train_config
  config_text = text_format.MessageToString(train_config, as_utf8=True)
  config_text = re.compile('decay_steps: \d+').\
      sub('decay_steps: %s' % feature_info_map['decay_steps']['decay_steps'], config_text)
  text_format.Merge(config_text, train_config)

  config_dir, config_name = os.path.split(FLAGS.output_config_path)
  config_util.save_pipeline_config(pipeline_config, config_dir, config_name)


if __name__ == '__main__':
  tf.app.run()
