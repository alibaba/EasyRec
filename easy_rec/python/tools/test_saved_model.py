# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

lookup_lib_path = 'libs/libkv_lookup.so'
lookup_op = tf.load_op_library(lookup_lib_path)

if __name__ == '__main__':
  """Test saved model, an example:

  python -m easy_rec.python.tools.test_saved_model
      --saved_model_dir after_edit_save
      --test_data_path data/test/rtp/xys_cxr_fg_sample_test2_with_lbl.txt
      --with_lbl
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--saved_model_dir', type=str, default=None, help='saved model dir')
  parser.add_argument(
      '--test_data_path', type=str, default=None, help='output dir')
  parser.add_argument(
      '--with_lbl',
      action='store_true',
      default=False,
      help='whether the test data has label field')
  args = parser.parse_args()

  logging.info('saved_model_dir: %s' % args.saved_model_dir)
  logging.info('test_data_path: %s' % args.test_data_path)
  logging.info('test_data has lbl: %s' % args.with_lbl)

  with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, ['serve'],
                                                args.saved_model_dir)
    signature_def = meta_graph_def.signature_def[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    graph = tf.get_default_graph()
    inputs = signature_def.inputs
    inputs_map = {}
    for name, tensor in inputs.items():
      logging.info('inputs: %s => %s' % (name, tensor.name))
      inputs_map[name] = graph.get_tensor_by_name(tensor.name)
    outputs = signature_def.outputs
    outputs_map = {}
    for name, tensor in outputs.items():
      logging.info('outputs: %s => %s' % (name, tensor.name))
      outputs_map[name] = graph.get_tensor_by_name(tensor.name)
    with open(args.test_data_path, 'r') as fin:
      feature_vals = []
      for line_str in fin:
        line_str = line_str.strip()
        line_toks = line_str.split('')
        if args.with_lbl:
          line_toks = line_toks[1:]
        feature_vals.append(''.join(line_toks))
        if len(feature_vals) >= 128:
          break
      out_vals = sess.run(
          outputs_map, feed_dict={inputs_map['features']: feature_vals})
      logging.info(out_vals)
