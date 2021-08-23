# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging
import os
import sys

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io.file_io import file_exists
from tensorflow.python.lib.io.file_io import recursive_create_dir
from tensorflow.python.platform.gfile import GFile

import easy_rec
from easy_rec.python.utils.meta_graph_editor import MetaGraphEditor

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
  """Replace the default embedding_lookup ops with self defined embedding lookup ops.

  The data are now stored in redis, for lookup, it is to retrieve the
  embedding vectors by {version}_{embed_name}_{embed_id}.
  Example:
    python -m easy_rec.python.tools.edit_lookup_graph
      --saved_model_dir rtp_large_embedding_export/1604304644
      --output_dir ./after_edit_save
      --test_data_path data/test/rtp/xys_cxr_fg_sample_test2_with_lbl.txt
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--saved_model_dir', type=str, default=None, help='saved model dir')
  parser.add_argument('--output_dir', type=str, default=None, help='output dir')
  parser.add_argument(
      '--redis_url', type=str, default='127.0.0.1:6379', help='redis url')
  parser.add_argument(
      '--redis_passwd', type=str, default='', help='redis password')
  parser.add_argument('--time_out', type=int, default=1500, help='timeout')
  parser.add_argument(
      '--test_data_path', type=str, default='', help='test data path')
  parser.add_argument('--verbose', action='store_true', default=False)

  args = parser.parse_args()
  logging.info('saved_model_dir: %s' % args.saved_model_dir)

  if not os.path.exists(os.path.join(args.saved_model_dir, 'saved_model.pb')):
    logging.error('saved_model.pb does not exist in %s' % args.saved_model_dir)
    sys.exit(1)

  logging.info('output_dir: %s' % args.output_dir)
  logging.info('redis_url: %s' % args.redis_url)
  lookup_lib_path = os.path.join(easy_rec.ops_dir, 'libkv_lookup.so')
  logging.info('lookup_lib_path: %s' % lookup_lib_path)

  if not file_exists(args.output_dir):
    recursive_create_dir(args.output_dir)

  meta_graph_editor = MetaGraphEditor(
      lookup_lib_path,
      args.saved_model_dir,
      args.redis_url,
      args.redis_passwd,
      args.time_out,
      meta_graph_def=None,
      debug_dir=args.output_dir if args.verbose else '')
  meta_graph_editor.edit_graph()

  meta_graph_version = meta_graph_editor.meta_graph_version
  if meta_graph_version == '':
    export_ts = [
        x for x in args.saved_model_dir.split('/') if x != '' and x is not None
    ]
    meta_graph_version = export_ts[-1]

  # import edit graph
  tf.reset_default_graph()
  saver = tf.train.import_meta_graph(meta_graph_editor._meta_graph_def)

  embed_name_to_id_file = os.path.join(args.output_dir, 'embed_name_to_ids.txt')
  with GFile(embed_name_to_id_file, 'w') as fout:
    for tmp_norm_name in meta_graph_editor._embed_name_to_ids:
      fout.write(
          '%s\t%s\n' %
          (tmp_norm_name, meta_graph_editor._embed_name_to_ids[tmp_norm_name]))
  tf.add_to_collection(
      tf.GraphKeys.ASSET_FILEPATHS,
      tf.constant(
          embed_name_to_id_file, dtype=tf.string, name='embed_name_to_ids.txt'))

  graph = tf.get_default_graph()
  inputs = meta_graph_editor.signature_def.inputs
  inputs_map = {}
  for name, tensor in inputs.items():
    logging.info('model inputs: %s => %s' % (name, tensor.name))
    inputs_map[name] = graph.get_tensor_by_name(tensor.name)

  outputs = meta_graph_editor.signature_def.outputs
  outputs_map = {}
  for name, tensor in outputs.items():
    logging.info('model outputs: %s => %s' % (name, tensor.name))
    outputs_map[name] = graph.get_tensor_by_name(tensor.name)
  with tf.Session() as sess:
    saver.restore(sess, args.saved_model_dir + '/variables/variables')
    output_dir = os.path.join(args.output_dir, meta_graph_version)
    tf.saved_model.simple_save(
        sess, output_dir, inputs=inputs_map, outputs=outputs_map)
    # the meta_graph_version could not be passed via existing interfaces
    # so we could only write it by the raw methods
    saved_model = saved_model_pb2.SavedModel()
    with GFile(os.path.join(output_dir, 'saved_model.pb'), 'rb') as fin:
      saved_model.ParseFromString(fin.read())

    saved_model.meta_graphs[
        0].meta_info_def.meta_graph_version = meta_graph_editor.meta_graph_version
    with GFile(os.path.join(output_dir, 'saved_model.pb'), 'wb') as fout:
      fout.write(saved_model.SerializeToString())

    logging.info('save output to %s' % output_dir)
    if args.test_data_path:
      with GFile(args.test_data_path, 'r') as fin:
        feature_vals = []
        for line_str in fin:
          line_str = line_str.strip()
          line_toks = line_str.split('')
          line_toks = line_toks[1:]
          feature_vals.append(''.join(line_toks))
          if len(feature_vals) >= 32:
            break
        out_vals = sess.run(
            outputs_map, feed_dict={inputs_map['features']: feature_vals})
        logging.info('test_data probs:' + str(out_vals))
