# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

import common_io
import tensorflow as tf

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
tf.app.flags.DEFINE_string('tables', '', 'quantile binning table')

FLAGS = tf.app.flags.FLAGS


def main(argv):
  pipeline_config = config_util.get_configs_from_pipeline_file(
      FLAGS.template_config_path)

  feature_boundaries_info = {}
  reader = common_io.table.TableReader(
      FLAGS.tables, selected_cols='feature,json')
  while True:
    try:
      record = reader.read()
      raw_info = json.loads(record[0][1])
      bin_info = []
      for info in raw_info['bin']['norm'][:-1]:
        split_point = float(info['value'].split(',')[1][:-1])
        bin_info.append(split_point)
      feature_boundaries_info[record[0][0]] = bin_info
    except common_io.exception.OutOfRangeException:
      reader.close()
      break

  logging.info('feature boundaries: %s' % feature_boundaries_info)

  for feature_config in pipeline_config.feature_configs:
    feature_name = feature_config.input_names[0]
    if feature_name in feature_boundaries_info:
      feature_config.feature_type = feature_config.RawFeature
      feature_config.hash_bucket_size = 0
      feature_config.boundaries.extend(feature_boundaries_info[feature_name])
      logging.info('edited %s' % feature_name)

  config_dir, config_name = os.path.split(FLAGS.output_config_path)
  config_util.save_pipeline_config(pipeline_config, config_dir, config_name)


if __name__ == '__main__':
  tf.app.run()
