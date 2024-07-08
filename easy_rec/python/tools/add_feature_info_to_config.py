# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

import tensorflow as tf

from easy_rec.python.utils import config_util
from easy_rec.python.utils.hive_utils import HiveUtils

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
  sels = 'feature,feature_info,message'
  feature_info_map = {}
  drop_feature_names = []

  if pipeline_config.WhichOneof('train_path') == 'hive_train_input':
    hive_util = HiveUtils(
        data_config=pipeline_config.data_config,
        hive_config=pipeline_config.hive_train_input,
        selected_cols=sels,
        record_defaults=['', '', ''])
    reader = hive_util.hive_read_line(FLAGS.config_table)
    for record in reader:
      feature_name = record[0][0]
      feature_info_map[feature_name] = json.loads(record[0][1])
      if 'DROP IT' in record[0][2]:
        drop_feature_names.append(feature_name)

  else:
    import common_io
    reader = common_io.table.TableReader(FLAGS.config_table, selected_cols=sels)
    while True:
      try:
        record = reader.read()
        feature_name = record[0][0]
        feature_info_map[feature_name] = json.loads(record[0][1])
        if 'DROP IT' in record[0][2]:
          drop_feature_names.append(feature_name)
      except common_io.exception.OutOfRangeException:
        reader.close()
        break

  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
  if drop_feature_names:
    tmp_feature_configs = feature_configs[:]
    for fea_cfg in tmp_feature_configs:
      fea_name = fea_cfg.input_names[0]
      if fea_name in drop_feature_names:
        feature_configs.remove(fea_cfg)
  for feature_config in feature_configs:
    feature_name = feature_config.input_names[0]
    if feature_name in feature_info_map:
      logging.info('edited %s' % feature_name)
      feature_config.embedding_dim = int(
          feature_info_map[feature_name]['embedding_dim'])
      logging.info('modify embedding_dim to %s' % feature_config.embedding_dim)
      if 'boundary' in feature_info_map[feature_name]:
        feature_config.ClearField('boundaries')
        feature_config.boundaries.extend(
            [float(i) for i in feature_info_map[feature_name]['boundary']])
        logging.info('modify boundaries to %s' % feature_config.boundaries)
      elif 'hash_bucket_size' in feature_info_map[feature_name]:
        feature_config.hash_bucket_size = int(
            feature_info_map[feature_name]['hash_bucket_size'])
        logging.info('modify hash_bucket_size to %s' %
                     feature_config.hash_bucket_size)
  # modify num_steps
  pipeline_config.train_config.num_steps = feature_info_map['__NUM_STEPS__'][
      'num_steps']
  logging.info('modify num_steps to %s' %
               pipeline_config.train_config.num_steps)
  # modify decay_steps
  optimizer_configs = pipeline_config.train_config.optimizer_config
  for optimizer_config in optimizer_configs:
    optimizer = optimizer_config.WhichOneof('optimizer')
    optimizer = getattr(optimizer_config, optimizer)
    learning_rate = optimizer.learning_rate.WhichOneof('learning_rate')
    learning_rate = getattr(optimizer.learning_rate, learning_rate)
    if hasattr(learning_rate, 'decay_steps'):
      learning_rate.decay_steps = feature_info_map['__DECAY_STEPS__'][
          'decay_steps']
    logging.info('modify decay_steps to %s' % learning_rate.decay_steps)

  for feature_group in pipeline_config.model_config.feature_groups:
    feature_names = feature_group.feature_names
    reserved_features = []
    for feature_name in feature_names:
      if feature_name not in drop_feature_names:
        reserved_features.append(feature_name)
      else:
        logging.info('drop feature: %s' % feature_name)
    feature_group.ClearField('feature_names')
    feature_group.feature_names.extend(reserved_features)
    for sequence_feature in feature_group.sequence_features:
      seq_att_maps = sequence_feature.seq_att_map
      for seq_att in seq_att_maps:
        keys = seq_att.key
        reserved_keys = []
        for key in keys:
          if key not in drop_feature_names:
            reserved_keys.append(key)
          else:
            logging.info('drop sequence feature key: %s' % key)
        seq_att.ClearField('key')
        seq_att.key.extend(reserved_keys)

        hist_seqs = seq_att.hist_seq
        reserved_hist_seqs = []
        for hist_seq in hist_seqs:
          if hist_seq not in drop_feature_names:
            reserved_hist_seqs.append(hist_seq)
          else:
            logging.info('drop sequence feature hist_seq: %s' % hist_seq)
        seq_att.ClearField('hist_seq')
        seq_att.hist_seq.extend(reserved_hist_seqs)

  config_dir, config_name = os.path.split(FLAGS.output_config_path)
  config_util.save_pipeline_config(pipeline_config, config_dir, config_name)


if __name__ == '__main__':
  tf.app.run()
