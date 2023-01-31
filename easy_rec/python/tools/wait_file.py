# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import os
import sys
import time

from tensorflow.python.platform import gfile

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-export_dir', type=str, default=None, help='saved model path')
  parser.add_argument(
      '-path_pattern',
      type=str,
      default='*/*/saved_model.pb',
      help='match path pattern')
  parser.add_argument(
      '-check_interval', type=int, default=10, help='check interval')
  parser.add_argument(
      '-max_wait_ts',
      type=int,
      default=24 * 3600,
      help='max wait time in seconds')
  args, _ = parser.parse_known_args()

  logging.info('export_dir=%s' % args.export_dir)
  logging.info('path_pattern=%s' % args.path_pattern)
  logging.info('check_interval=%d' % args.check_interval)
  logging.info('max_wait_ts=%d' % args.max_wait_ts)

  start_ts = time.time()
  saved_model_path = None
  search_pattern = os.path.join(args.export_dir, args.path_pattern)
  while time.time() - start_ts < args.max_wait_ts:
    # export/large
    saved_models = gfile.Glob(search_pattern)
    if len(saved_models) > 0:
      saved_model_path = saved_models[0]
      logging.info('find saved_model: %s' % saved_model_path)
      break
    else:
      time.sleep(args.check_interval)
  if saved_model_path is None:
    logging.error('failed to find saved_model in %s' % args.export_dir)
    sys.exit(1)
