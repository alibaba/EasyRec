# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import logging
import sys

import numpy as np

from easy_rec.python.inference.predictor import Predictor

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
  parser.add_argument('--tol', type=float, default=1e-5, help='tolerance')
  parser.add_argument('--separator', type=str, default='', help='separator')
  args = parser.parse_args()

  if not args.saved_model_dir:
    logging.error('saved_model_dir is not set')
    sys.exit(1)

  if not args.input_path:
    logging.error('input_path is not set')
    sys.exit(1)

  logging.info('input_path: ' + args.input_path)
  logging.info('save_path: ' + args.save_path)
  logging.info('separator: ' + args.separator)

  predictor = Predictor(args.saved_model_dir)
  with open(args.input_path, 'r') as fin:
    batch_input = []
    for line_str in fin:
      line_str = line_str.strip()
      line_tok = line_str.split(args.separator)
      feature = line_tok[-1]
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
