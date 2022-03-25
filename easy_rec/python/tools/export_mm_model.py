# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import signature_constants

from easy_rec.python.main import export
from easy_rec.python.utils import config_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', '', '')
tf.app.flags.DEFINE_string('pipeline_config_path', None, '')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'checkpoint to be exported')
tf.app.flags.DEFINE_string('rec_model_export_dir', None,
                           'directory where rec model should be exported to')
tf.app.flags.DEFINE_string('total_model_export_dir', None,
                           'directory where total model should be exported to')
tf.app.flags.DEFINE_string('img_model_export_dir', None,
                           'directory where img model should be exported to')
tf.app.flags.DEFINE_string('asset_files', '', 'more files to add to asset')
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


def cut_model(input_savemodel_path, output_img_savemodel_path,
              img_path_fea_name):
  model_dir = search_pb(input_savemodel_path)

  graph = tf.Graph()
  session_config = tf.ConfigProto()
  session_config.log_device_placement = True
  session_config.allow_soft_placement = True
  session_config.intra_op_parallelism_threads = 10
  session_config.inter_op_parallelism_threads = 10
  session_config.gpu_options.allow_growth = True

  with tf.Session(config=session_config, graph=graph) as sess:

    def device_func(o):
      return '/device:CPU:0'

    with tf.device(device_func):
      meta_graph_def = tf.saved_model.loader.load(
          sess, [tf.saved_model.tag_constants.SERVING], model_dir)

    input_map_names = {}
    output_map_names = {}

    signature_def = meta_graph_def.signature_def[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    inputs = signature_def.inputs
    for name, tensor in inputs.items():
      if name in [img_path_fea_name]:
        tensor = graph.get_tensor_by_name(tensor.name)
        input_map_names[name] = tensor

    outputs = signature_def.outputs
    for name, tensor in outputs.items():
      if name in ['img_emb']:
        tensor = graph.get_tensor_by_name(tensor.name)
        output_map_names[name] = tensor

    inputs = {}
    for k, v in input_map_names.items():
      inputs[k] = tf.saved_model.utils.build_tensor_info(v)
    outputs = {}
    for k, v in output_map_names.items():
      outputs[k] = tf.saved_model.utils.build_tensor_info(v)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder = tf.saved_model.builder.SavedModelBuilder(
        output_img_savemodel_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
        })
    builder.save()


def main(argv):

  assert FLAGS.model_dir or FLAGS.pipeline_config_path, 'At least one of model_dir and pipeline_config_path exists.'
  if FLAGS.model_dir:
    pipeline_config_path = os.path.join(FLAGS.model_dir, 'pipeline.config')
    if file_io.file_exists(pipeline_config_path):
      logging.info('update pipeline_config_path to %s' % pipeline_config_path)
    else:
      pipeline_config_path = FLAGS.pipeline_config_path
  else:
    pipeline_config_path = FLAGS.pipeline_config_path

  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
  total_model_pipeline_filename = 'pipeline_total_model.config'
  total_model_pipeline_path = os.path.join(FLAGS.model_dir,
                                           total_model_pipeline_filename)
  rec_model_pipeline_filename = 'pipeline_rec_model.config'
  rec_model_pipeline_path = os.path.join(FLAGS.model_dir,
                                         rec_model_pipeline_filename)

  # step 1 : modify config for total model
  drop_idx = None
  sample_num_fea_name = None
  for idx, fea_config in enumerate(feature_configs):
    if fea_config.feature_type == fea_config.SampleNumFeature:
      sample_num_fea_name = fea_config.input_names[0]
      drop_idx = idx
      break
  del feature_configs[drop_idx]
  assert sample_num_fea_name is not None, 'SampleNumFeature has not be set in %s.' % pipeline_config_path

  drop_idx = None
  for idx, input_config in enumerate(pipeline_config.data_config.input_fields):
    if input_config.input_name == sample_num_fea_name:
      drop_idx = idx
      break
  del pipeline_config.data_config.input_fields[drop_idx]

  drop_idx = None
  for idx, fea_group_config in enumerate(
      pipeline_config.model_config.feature_groups):
    if len(fea_group_config.feature_names
           ) == 1 and fea_group_config.feature_names[0] == sample_num_fea_name:
      drop_idx = idx
  del pipeline_config.model_config.feature_groups[drop_idx]

  config_util.save_pipeline_config(
      pipeline_config, FLAGS.model_dir, filename=total_model_pipeline_filename)

  # step 2 : export total model
  export(FLAGS.total_model_export_dir, total_model_pipeline_path,
         FLAGS.checkpoint_path, FLAGS.asset_files)

  # step 3 : modify config for rec model
  model_config = pipeline_config.model_config.e2e_mm_dbmtl
  if model_config.HasField('highway_dnn'):
    emb_size = model_config.highway_dnn.emb_size
  elif model_config.HasField('img_dnn'):
    emb_size = model_config.img_dnn.hidden_units[-1]
  else:
    emb_size = model_config.img_model.num_classes

  img_path_fea_name = None
  for idx, fea_config in enumerate(feature_configs):
    if fea_config.feature_type == fea_config.ImgFeature:
      img_path_fea_name = fea_config.input_names[0]
      fea_config.input_names[0] = 'img_emb'
      fea_config.feature_type = fea_config.RawFeature
      fea_config.separator = ','
      fea_config.raw_input_dim = emb_size

  assert img_path_fea_name is not None, 'ImgFeature has not be set in %s.' % pipeline_config_path

  for idx, input_config in enumerate(pipeline_config.data_config.input_fields):
    if input_config.input_name == img_path_fea_name:
      input_config.input_name = 'img_emb'
      break

  for idx, fea_group_config in enumerate(
      pipeline_config.model_config.feature_groups):
    if fea_group_config.group_name == 'img':
      fea_group_config.group_name = 'img_emb'
      fea_group_config.feature_names[0] = 'img_emb'
  config_util.save_pipeline_config(
      pipeline_config, FLAGS.model_dir, filename=rec_model_pipeline_filename)

  # step 4 : export rec model
  export(FLAGS.rec_model_export_dir, rec_model_pipeline_path,
         FLAGS.checkpoint_path, FLAGS.asset_files)

  # step 5 : cut img model from all model
  cut_model(FLAGS.total_model_export_dir, FLAGS.img_model_export_dir,
            img_path_fea_name)


if __name__ == '__main__':
  tf.app.run()
