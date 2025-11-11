# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import logging
import sys

try:
  from easy_rec.python.utils.io_util import http_read
except Exception as ex:
  logging.error(ex)
  sys.exit(2)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--pull_request_num', type=int, default=None, help='pull request number')
  parser.add_argument('--exclude_dirs', nargs='*', type=str, help='the directory to be ignored')

  args = parser.parse_args()

  url = 'https://api.github.com/repos/alibaba/EasyRec/pulls/%d/files' % args.pull_request_num
  pull_request_data = http_read(url)

  changes = json.loads(pull_request_data)
  change_dir = []
  for obj in changes:
    filename = obj['filename']
    toks = filename.split('/')
    if len(toks) > 0:
      if toks[0] not in args.exclude_dirs:
        change_dir.append(toks[0])

  change_dir = list(set(change_dir))
  logging.info('changed directories: %s' % ','.join(change_dir))
  if len(change_dir) == 0:
    sys.exit(1)
