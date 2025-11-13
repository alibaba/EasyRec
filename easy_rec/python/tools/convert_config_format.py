# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from google.protobuf import json_format
from google.protobuf import text_format

from easy_rec.python.protos.pipeline_pb2 import EasyRecConfig


def load_config(input_config):
  pipeline_config = EasyRecConfig()
  with open(input_config, 'r') as fin:
    tmp_str = fin.read()
    if input_config.endswith('.config'):
      text_format.Merge(tmp_str, pipeline_config)
    elif input_config.endswith('.json'):
      json_format.Parse(tmp_str, pipeline_config)
    else:
      assert False, 'only .config/.json are supported(%s)' % input_config
  return pipeline_config


def save_config(pipeline_config, save_path):
  with open(save_path, 'w') as fout:
    if save_path.endswith('.config'):
      fout.write(text_format.MessageToString(pipeline_config, as_utf8=True))
    elif save_path.endswith('.json'):
      fout.write(
          json_format.MessageToJson(
              pipeline_config, preserving_proto_field_name=True))
    else:
      assert False, 'only .config/.json are supported(%s)' % save_path


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_config', type=str, help='input_config path', default=None)
  parser.add_argument(
      '--output_config', type=str, help='output_config path', default=None)
  args = parser.parse_args()

  assert os.path.exists(args.input_config)
  pipeline_config = load_config(args.input_config)
  save_config(pipeline_config, args.output_config)
