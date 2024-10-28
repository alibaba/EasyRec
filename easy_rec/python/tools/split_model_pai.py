# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework.dtypes import _TYPE_TO_STRING
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import saver as tf_saver

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
  from tensorflow.python.saved_model.path_helpers import get_variables_path
  from tensorflow.python.ops.resource_variable_ops import _from_proto_fn
else:
  from tensorflow.python.saved_model.utils_impl import get_variables_path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', '', '')
tf.app.flags.DEFINE_string('user_model_dir', '', '')
tf.app.flags.DEFINE_string('item_model_dir', '', '')
tf.app.flags.DEFINE_string('user_fg_json_path', '', '')
tf.app.flags.DEFINE_string('item_fg_json_path', '', '')

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def search_pb(directory):
  dir_list = []
  for root, dirs, files in tf.gfile.Walk(directory):
    for f in files:
      _, ext = os.path.splitext(f)
      if ext == '.pb':
        dir_list.append(root)
  if len(dir_list) == 0:
    raise ValueError('savedmodel is not found in directory %s' % directory)
  elif len(dir_list) > 1:
    raise ValueError('multiple saved model found in directory %s' % directory)

  return dir_list[0]


def _node_name(name):
  if name.startswith('^'):
    return name[1:]
  else:
    return name.split(':')[0]


def extract_sub_graph(graph_def, dest_nodes, variable_protos):
  """Extract the subgraph that can reach any of the nodes in 'dest_nodes'.

  Args:
      graph_def: graph_pb2.GraphDef
      dest_nodes: a list includes output node names

  Returns:
      out: the GraphDef of the sub-graph.
      variables_to_keep: variables to be kept for saver.
  """
  if not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError('graph_def must be a graph_pb2.GraphDef proto.')

  edges = {}
  name_to_node_map = {}
  node_seq = {}
  seq = 0
  nodes_to_keep = set()
  variables_to_keep = set()

  for node in graph_def.node:
    n = _node_name(node.name)
    name_to_node_map[n] = node
    edges[n] = [_node_name(item) for item in node.input]
    node_seq[n] = seq
    seq += 1
  for d in dest_nodes:
    assert d in name_to_node_map, "'%s' is not in graph" % d

  next_to_visit = dest_nodes[:]
  while next_to_visit:
    n = next_to_visit[0]

    if n in variable_protos:
      proto = variable_protos[n]
      next_to_visit.append(_node_name(proto.initial_value_name))
      next_to_visit.append(_node_name(proto.initializer_name))
      next_to_visit.append(_node_name(proto.snapshot_name))
      variables_to_keep.add(proto.variable_name)

    del next_to_visit[0]
    if n in nodes_to_keep:
      continue
    # make sure n is in edges
    if n in edges:
      nodes_to_keep.add(n)
      next_to_visit += edges[n]
  nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

  out = graph_pb2.GraphDef()
  for n in nodes_to_keep_list:
    out.node.extend([copy.deepcopy(name_to_node_map[n])])
  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)

  return out, variables_to_keep


def load_meta_graph_def(model_dir):
  """Load meta graph def in saved model.

  Args:
      model_dir: saved model directory.

  Returns:
      meta_graph_def: a MetaGraphDef.
      variable_protos: a dict of VariableDef.
      input_tensor_names: signature inputs in saved model.
      output_tensor_names: signature outputs in saved model.
  """
  input_tensor_names = {}
  output_tensor_names = {}
  variable_protos = {}

  meta_graph_def = saved_model_utils.get_meta_graph_def(
      model_dir, tf.saved_model.tag_constants.SERVING)
  signatures = meta_graph_def.signature_def
  collections = meta_graph_def.collection_def

  # parse collection_def in SavedModel
  for key, col_def in collections.items():
    if key in ops.GraphKeys._VARIABLE_COLLECTIONS:
      tf.logging.info('[Collection] %s:' % key)
      for value in col_def.bytes_list.value:
        proto_type = ops.get_collection_proto_type(key)
        proto = proto_type()
        proto.ParseFromString(value)
        tf.logging.info('%s' % proto.variable_name)
        variable_node_name = _node_name(proto.variable_name)
        if variable_node_name not in variable_protos:
          variable_protos[variable_node_name] = proto

  # parse signature info for SavedModel
  for sig_name in signatures:
    if signatures[
        sig_name].method_name == tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
      tf.logging.info('[Signature] inputs:')
      for input_name in signatures[sig_name].inputs:
        input_tensor_shape = []
        input_tensor = signatures[sig_name].inputs[input_name]
        for dim in input_tensor.tensor_shape.dim:
          input_tensor_shape.append(int(dim.size))
        tf.logging.info('"%s": %s; %s' %
                        (input_name, _TYPE_TO_STRING[input_tensor.dtype],
                         input_tensor_shape))
        input_tensor_names[input_name] = input_tensor.name
      tf.logging.info('[Signature] outputs:')
      for output_name in signatures[sig_name].outputs:
        output_tensor_shape = []
        output_tensor = signatures[sig_name].outputs[output_name]
        for dim in output_tensor.tensor_shape.dim:
          output_tensor_shape.append(int(dim.size))
        tf.logging.info('"%s": %s; %s' %
                        (output_name, _TYPE_TO_STRING[output_tensor.dtype],
                         output_tensor_shape))
        output_tensor_names[output_name] = output_tensor.name

  return meta_graph_def, variable_protos, input_tensor_names, output_tensor_names


def export(model_dir, meta_graph_def, variable_protos, input_tensor_names,
           output_tensor_names, part_name, part_dir):
  """Export subpart saved model.

  Args:
      model_dir: saved model directory.
      meta_graph_def: a MetaGraphDef.
      variable_protos: a dict of VariableDef.
      input_tensor_names: signature inputs in saved model.
      output_tensor_names: signature outputs in saved model.
      part_name: subpart model name, user or item.
      part_dir: subpart model export directory.
  """
  output_tensor_names = {
      x: output_tensor_names[x]
      for x in output_tensor_names.keys()
      if part_name in x
  }
  output_node_names = [
      _node_name(output_tensor_names[x]) for x in output_tensor_names.keys()
  ]

  inference_graph, variables_to_keep = extract_sub_graph(
      meta_graph_def.graph_def, output_node_names, variable_protos)

  tf.reset_default_graph()
  with tf.Session() as sess:
    with sess.graph.as_default():
      graph = ops.get_default_graph()
      importer.import_graph_def(inference_graph, name='')
      for name in variables_to_keep:
        if tf.__version__ >= '2.0':
          variable = _from_proto_fn(variable_protos[name.split(':')[0]])
        else:
          variable = graph.get_tensor_by_name(name)
        graph.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, variable)
      saver = tf_saver.Saver()
      saver.restore(sess, get_variables_path(model_dir))

      builder = tf.saved_model.builder.SavedModelBuilder(part_dir)
      signature_inputs = {}
      for input_name in input_tensor_names:
        try:
          tensor_info = tf.saved_model.utils.build_tensor_info(
              graph.get_tensor_by_name(input_tensor_names[input_name]))
          signature_inputs[input_name] = tensor_info
        except Exception:
          print('ignore input: %s' % input_name)

      signature_outputs = {}
      for output_name in output_tensor_names:
        tensor_info = tf.saved_model.utils.build_tensor_info(
            graph.get_tensor_by_name(output_tensor_names[output_name]))
        signature_outputs[output_name] = tensor_info

      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs=signature_inputs,
              outputs=signature_outputs,
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))

      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  prediction_signature,
          })
      builder.save()
  config_path = os.path.join(model_dir, 'assets/pipeline.config')
  assert tf.gfile.Exists(config_path)
  dst_path = os.path.join(part_dir, 'assets')
  dst_config_path = os.path.join(dst_path, 'pipeline.config')
  tf.gfile.MkDir(dst_path)
  tf.gfile.Copy(config_path, dst_config_path)
  if part_name == 'user' and FLAGS.user_fg_json_path:
    dst_fg_path = os.path.join(dst_path, 'fg.json')
    tf.gfile.Copy(FLAGS.user_fg_json_path, dst_fg_path)
  if part_name == 'item' and FLAGS.item_fg_json_path:
    dst_fg_path = os.path.join(dst_path, 'fg.json')
    tf.gfile.Copy(FLAGS.item_fg_json_path, dst_fg_path)


def main(argv):
  model_dir = search_pb(FLAGS.model_dir)
  tf.logging.info('Loading meta graph...')
  meta_graph_def, variable_protos, input_tensor_names, output_tensor_names = load_meta_graph_def(
      model_dir)
  tf.logging.info('Exporting user part model...')
  export(
      model_dir,
      meta_graph_def,
      variable_protos,
      input_tensor_names,
      output_tensor_names,
      part_name='user',
      part_dir=FLAGS.user_model_dir)
  tf.logging.info('Exporting item part model...')
  export(
      model_dir,
      meta_graph_def,
      variable_protos,
      input_tensor_names,
      output_tensor_names,
      part_name='item',
      part_dir=FLAGS.item_model_dir)


if __name__ == '__main__':
  tf.app.run()
