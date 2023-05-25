# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging

from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.platform.gfile import GFile

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', type=str, default=None, help='saved model path')
  parser.add_argument(
      '--output', type=str, default=None, help='saved model save path')
  args = parser.parse_args()

  assert args.input is not None and args.output is not None

  logging.info('saved_model_path: %s' % args.input)

  saved_model = saved_model_pb2.SavedModel()
  if args.input.endswith('.pb'):
    with GFile(args.input, 'rb') as fin:
      saved_model.ParseFromString(fin.read())
  else:
    with GFile(args.input, 'r') as fin:
      text_format.Merge(fin.read(), saved_model)

  if args.output.endswith('.pbtxt'):
    with GFile(args.output, 'w') as fout:
      fout.write(text_format.MessageToString(saved_model, as_utf8=True))
  else:
    with GFile(args.output, 'wb') as fout:
      fout.write(saved_model.SerializeToString())
