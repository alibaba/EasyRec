# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import logging
import os
import sys

import numpy as np

import easy_rec
from easy_rec.python.inference.predictor import Predictor

try:
  import tensorflow as tf
  tf.load_op_library(os.path.join(easy_rec.ops_dir, 'libembed_op.so'))
except Exception as ex:
  logging.warning('exception: %s' % str(ex))

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--saved_model_dir', type=str, default=None, help='saved model directory')
  parser.add_argument(
      '--input_path', type=str, default=None, help='input feature path')
  parser.add_argument('--save_path', type=str, default=None, help='save path')
  parser.add_argument(
      '--cmp_res_path', type=str, default=None, help='compare result path')
  parser.add_argument(
      '--cmp_key', type=str, default='probs', help='compare key')
  parser.add_argument(
      '--rtp_fea_id',
      type=int,
      default=-1,
      help='rtp feature column index, default to the last column')
  parser.add_argument('--tol', type=float, default=1e-5, help='tolerance')
  parser.add_argument(
      '--label_id',
      nargs='*',
      type=int,
      help='the label column, which is to be excluded')
  parser.add_argument(
      '--separator',
      type=str,
      default='',
      help='separator between features, default to \\u0002')
  parser.add_argument(
      '--rtp_separator',
      type=str,
      default='',
      help='separator, default to \\u0001')
  args = parser.parse_args()

  if not args.saved_model_dir:
    logging.error('saved_model_dir is not set')
    sys.exit(1)

  if not args.input_path:
    logging.error('input_path is not set')
    sys.exit(1)

  if args.label_id is None:
    args.label_id = []

  logging.info('input_path: ' + args.input_path)
  logging.info('save_path: ' + args.save_path)
  logging.info('separator: ' + args.separator)

  predictor = Predictor(args.saved_model_dir)
  if len(predictor.input_names) == 1:
    assert len(
        args.label_id
    ) == 0, 'label_id should not be set if rtp feature format is used.'

  with open(args.input_path, 'r') as fin:
    batch_input = []
    for line_str in fin:
      line_str = line_str.strip()
      line_tok = line_str.split(args.rtp_separator)
      feature = line_tok[args.rtp_fea_id]
      feature = [
          x for fid, x in enumerate(feature.split(args.separator))
          if fid not in args.label_id
      ]
      if 'features' in predictor.input_names:
        feature = args.separator.join(feature)
      batch_input.append(feature)
    output = predictor.predict(batch_input)

  if args.save_path:
    fout = open(args.save_path, 'w')
    for one in output:
      fout.write(str(one) + '\n')
    fout.close()

  if args.cmp_res_path:
    logging.info('compare result path: ' + args.cmp_res_path)
    logging.info('compare key: ' + args.cmp_key)
    logging.info('tolerance: ' + str(args.tol))
    with open(args.cmp_res_path, 'r') as fin:
      for line_id, line_str in enumerate(fin):
        line_str = line_str.strip()
        line_pred = json.loads(line_str)
        assert np.abs(
            line_pred[args.cmp_key] -
            output[line_id][args.cmp_key]) < args.tol, 'line[%d]: %.8f' % (
                line_id,
                np.abs(line_pred[args.cmp_key] - output[line_id][args.cmp_key]))
