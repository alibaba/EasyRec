# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convert rtp fg feature config to easy_rec data_config and feature_config."""
import argparse
import logging
import sys

import tensorflow as tf

from easy_rec.python.utils.config_util import save_message
from easy_rec.python.utils.convert_rtp_fg import convert_rtp_fg

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

model_types = ['deepfm', 'multi_tower', 'wide_and_deep', 'esmm', 'dbmtl', '']
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_type',
      type=str,
      choices=model_types,
      default='',
      help='model type, currently support: %s' % ','.join(model_types))
  parser.add_argument('--rtp_fg', type=str, help='rtp fg path')
  parser.add_argument(
      '--embedding_dim', type=int, default=16, help='embedding_dimension')
  parser.add_argument(
      '--batch_size', type=int, default=1024, help='batch_size for train')
  parser.add_argument(
      '--label',
      type=str,
      default='',
      nargs='+',
      required=True,
      help='label fields')
  parser.add_argument(
      '--num_steps',
      type=int,
      default=1000,
      help='number of train steps = num_samples * num_epochs / batch_size / num_workers'
  )
  parser.add_argument('--output_path', type=str, help='generated config path')
  parser.add_argument(
      '--incol_separator',
      type=str,
      default='\003',
      help='separator for multi_value features')
  parser.add_argument(
      '--separator',
      type=str,
      default='\002',
      help='separator between different features')
  parser.add_argument(
      '--train_input_path', type=str, default=None, help='train data path')
  parser.add_argument(
      '--eval_input_path', type=str, default=None, help='eval data path')
  parser.add_argument(
      '--selected_cols',
      type=str,
      default=None,
      help='selected cols, for csv input, it is in the format of: label_col_id0,...,lable_cold_idn,feature_col_id '
      'for odps table input, it is in the format of: label_col_name0,...,label_col_namen,feature_col_name '
  )
  parser.add_argument(
      '--rtp_separator', type=str, default=';', help='separator')
  parser.add_argument(
      '--input_type',
      type=str,
      default='OdpsRTPInput',
      help='default to OdpsRTPInput, if test local, change it to RTPInput')
  parser.add_argument(
      '--is_async', action='store_true', help='async mode, debug to false')

  args = parser.parse_args()

  if not args.rtp_fg:
    logging.error('rtp_fg is not set')
    sys.exit(1)

  if not args.output_path:
    logging.error('output_path is not set')
    sys.exit(1)

  pipeline_config = convert_rtp_fg(args.rtp_fg, args.embedding_dim,
                                   args.batch_size, args.label, args.num_steps,
                                   args.model_type, args.separator,
                                   args.incol_separator, args.train_input_path,
                                   args.eval_input_path, args.selected_cols,
                                   args.input_type, args.is_async)
  save_message(pipeline_config, args.output_path)
  logging.info('Conversion done.')
  logging.info('Tips:')
  logging.info(
      'if run on local, please change data_config.input_type to RTPInput, '
      'and model_dir/train_input_path/eval_input_path must also be set, ')
  logging.info(
      'if run local, please set data_config.selected_cols in the format '
      'label_col_id0,label_col_id1,...,label_col_idn,feature_col_id')
  logging.info(
      'if run on odps, selected_cols must be set, which are label0_col,'
      'label1_col, ..., feature_col_name')
