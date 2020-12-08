# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Convert the original rtp data format to csv format.

The original data format is not suggested to use with EasyRec.
In the original format: features are in kv format, if a feature has
more than one value, there will be multiple kvs, such as:
  ...tagbeautytagsmart...
In our new format:
  ...beautysmart...
"""
import argparse
import csv
import json
import logging
import sys

import tensorflow as tf

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--rtp_fg', type=str, default='', help='rtp fg path(.json)')
  parser.add_argument('--input_path', type=str, default='', help='input path')
  parser.add_argument('--output_path', type=str, default='', help='output path')
  parser.add_argument('--label', type=str, default='', help='label for train')
  args = parser.parse_args()

  if not args.rtp_fg:
    logging.error('rtp_fg is not set')
    sys.exit(1)

  if not args.input_path:
    logging.error('input_path is not set')
    sys.exit(1)

  if not args.output_path:
    logging.error('output_path is not set')
    sys.exit(1)

  if not args.label:
    logging.error('label is not set')
    sys.exit(1)

  with open(args.rtp_fg, 'r') as fin:
    rtp_fg = json.load(fin)

  feature_names = [args.label]
  for feature in rtp_fg['features']:
    feature_name = feature['feature_name']
    feature_names.append(feature_name)

  with open(args.input_path, 'r') as fin:
    with open(args.output_path, 'w') as fout:
      writer = csv.writer(fout)
      for line_str in fin:
        line_str = line_str.strip()
        line_toks = line_str.split('\002')
        temp_dict = {}
        for line_tok in line_toks:
          k, v = line_tok.split('\003')
          if k not in temp_dict:
            temp_dict[k] = [v]
          else:
            temp_dict[k].append(v)
        temp_vs = []
        for feature_name in feature_names:
          if feature_name in temp_dict:
            temp_vs.append('|'.join(temp_dict[feature_name]))
          else:
            temp_vs.append('')
        writer.writerow(temp_vs)
