# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf

import easy_rec
from easy_rec.python.inference.predictor import Predictor

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

lookup_op_path = os.path.join(easy_rec.ops_dir, 'libkv_lookup.so')
lookup_op = tf.load_op_library(lookup_op_path)

if __name__ == '__main__':
  """Test saved model, an example:

  python -m easy_rec.python.tools.test_saved_model
      --saved_model_dir after_edit_save
      --input_path data/test/rtp/xys_cxr_fg_sample_test2_with_lbl.txt
      --with_lbl
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--saved_model_dir', type=str, default=None, help='saved model dir')
  parser.add_argument('--input_path', type=str, default=None, help='output dir')
  parser.add_argument('--save_path', type=str, default=None, help='save path')
  parser.add_argument('--separator', type=str, default=',', help='separator')
  parser.add_argument(
      '--cmp_res_path', type=str, default=None, help='compare result path')
  parser.add_argument(
      '--cmp_key', type=str, default='probs', help='compare key')
  parser.add_argument('--tol', type=float, default=1e-5, help='tolerance')
  parser.add_argument(
      '--with_lbl',
      action='store_true',
      default=False,
      help='whether the test data has label field')
  args = parser.parse_args()

  logging.info('saved_model_dir: %s' % args.saved_model_dir)
  logging.info('test_data_path: %s' % args.input_path)
  logging.info('test_data has lbl: %s' % args.with_lbl)

  predictor = Predictor(args.saved_model_dir)
  with open(args.input_path, 'r') as fin:
    feature_vals = []
    for line_str in fin:
      line_str = line_str.strip()
      line_toks = line_str.split(args.separator)
      if args.with_lbl:
        line_toks = line_toks[1:]
      feature_vals.append(args.separator.join(line_toks))
  output = predictor.predict(feature_vals, batch_size=4096)

  if args.save_path:
    with open(args.save_path, 'w') as fout:
      for one in output:
        fout.write(str(one) + '\n')

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
