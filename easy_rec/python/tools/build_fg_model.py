import logging
import os
import json
import tensorflow as tf


curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
ops_idr = os.path.dirname(parent_dir)
ops_dir = os.path.join(ops_idr, 'ops')
if 'PAI' in tf.__version__:
  ops_dir = os.path.join(ops_dir, '1.12_pai')
elif tf.__version__.startswith('1.12'):
  ops_dir = os.path.join(ops_dir, '1.12')
elif tf.__version__.startswith('1.15'):
  if 'IS_ON_PAI' in os.environ:
    ops_dir = os.path.join(ops_dir, 'DeepRec')
  else:
    ops_dir = os.path.join(ops_dir, '1.15')
else:
  ops_dir = None


def load_fg_config(fg_json):
  with open(fg_json, 'r') as f:
    fg = json.load(f)
  features = fg['features']
  print(features[0])


if __name__ == '__main__':
  load_fg_config("fg.json")
