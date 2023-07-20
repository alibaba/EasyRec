# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import logging
import math
import os
import time

import numpy as np
import pandas as pd
import six
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import signature_constants

from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import numpy_utils
from easy_rec.python.utils import tf_utils
from easy_rec.python.utils.check_utils import check_split
from easy_rec.python.utils.config_util import get_configs_from_pipeline_file
from easy_rec.python.utils.config_util import get_input_name_from_fg_json
from easy_rec.python.utils.config_util import search_fg_json
from easy_rec.python.utils.hive_utils import HiveUtils
from easy_rec.python.utils.input_utils import get_type_defaults
from easy_rec.python.utils.load_class import get_register_class_meta
from easy_rec.python.utils.tf_utils import get_tf_type

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

SINGLE_PLACEHOLDER_FEATURE_KEY = 'features'

_PREDICTOR_CLASS_MAP = {}
_register_abc_meta = get_register_class_meta(
    _PREDICTOR_CLASS_MAP, have_abstract_class=True)


class PredictorInterface(six.with_metaclass(_register_abc_meta, object)):
  version = 1

  def __init__(self, model_path, model_config=None):
    """Init tensorflow session and load tf model.

    Args:
      model_path:  init model from this directory
      model_config: config string for model to init, in json format
    """
    pass

  @abc.abstractmethod
  def predict(self, input_data, batch_size):
    """Using session run predict a number of samples using batch_size.

    Args:
      input_data:  a list of numpy array, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime

    Returns:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
    pass

  def get_output_type(self):
    """Get output types of prediction.

    In this function user should return a type dict, which indicates which type of
    data should the output of predictor be converted to.

    In this function user should return a type dict, which indicates
    which type of data should the output of predictor be converted to
    * type json, data will be serialized to json str

    * type image, data will be converted to encode image binary and write to oss file,
      whose name is output_dir/${key}/${input_filename}_${idx}.jpg, where input_filename
      is extracted from url, key corresponds to the key in the dict of output_type,
      if the type of data indexed by key is a list, idx is the index of element in list, otherwhile ${idx} will be empty

    * type video, data will be converted to encode video binary and write to oss file,

    eg:  return  {
      'image': 'image',
      'feature': 'json'
    }

    indicating that the image data in the output dict will be save to image
    file and feature in output dict will be converted to json
    """
    return {}


class PredictorImpl(object):

  def __init__(self, model_path, profiling_file=None, use_latest=False):
    """Impl class for predictor.

    Args:
      model_path:  saved_model directory or frozenpb file path
      profiling_file:  profiling result file, default None.
        if not None, predict function will use Timeline to profiling
        prediction time, and the result json will be saved to profiling_file
      use_latest: use latest saved_model.pb if multiple ones are found,
        else raise an exception.
    """
    self._inputs_map = {}
    self._outputs_map = {}
    self._is_saved_model = False
    self._profiling_file = profiling_file
    self._model_path = model_path
    self._input_names = []
    self._is_multi_placeholder = True
    self._use_latest = use_latest

    self._build_model()

  @property
  def input_names(self):
    return self._input_names

  @property
  def output_names(self):
    return list(self._outputs_map.keys())

  def __del__(self):
    """Destroy predictor resources."""
    self._session.close()

  def search_pb(self, directory):
    """Search pb file recursively in model directory. if multiple pb files exist, exception will be raised.

    If multiple pb files exist, exception will be raised.

    Args:
      directory: model directory.

    Returns:
      directory contain pb file
    """
    dir_list = []
    for root, dirs, files in gfile.Walk(directory):
      for f in files:
        if f.endswith('saved_model.pb'):
          dir_list.append(root)
    if len(dir_list) == 0:
      raise ValueError('savedmodel is not found in directory %s' % directory)
    elif len(dir_list) > 1:
      if self._use_latest:
        logging.info('find %d models: %s' % (len(dir_list), ','.join(dir_list)))
        dir_list = sorted(
            dir_list,
            key=lambda x: int(x.split('/')[(-2 if (x[-1] == '/') else -1)]))
        return dir_list[-1]
      else:
        raise ValueError('multiple saved model found in directory %s' %
                         directory)

    return dir_list[0]

  def _get_input_fields_from_pipeline_config(self, model_path):
    pipeline_path = os.path.join(model_path, 'assets/pipeline.config')
    if not gfile.Exists(pipeline_path):
      logging.warning(
          '%s not exists, default values maybe inconsistent with the values used in training.'
          % pipeline_path)
      return {}
    pipeline_config = get_configs_from_pipeline_file(pipeline_path)
    input_fields = pipeline_config.data_config.input_fields
    input_fields_info = {
        input_field.input_name:
        (input_field.input_type, input_field.default_val)
        for input_field in input_fields
    }
    input_fields_list = [input_field.input_name for input_field in input_fields]

    return input_fields_info, input_fields_list

  def _build_model(self):
    """Load graph from model_path and create session for this graph."""
    model_path = self._model_path
    self._graph = tf.Graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=(self._profiling_file is not None))
    self._session = tf.Session(config=session_config, graph=self._graph)

    with self._graph.as_default():
      with self._session.as_default():
        # load model
        _, ext = os.path.splitext(model_path)
        tf.logging.info('loading model from %s' % model_path)
        if gfile.IsDirectory(model_path):
          model_path = self.search_pb(model_path)
          logging.info('model find in %s' % model_path)
          self._input_fields_info, self._input_fields_list = self._get_input_fields_from_pipeline_config(
              model_path)
          assert tf.saved_model.loader.maybe_saved_model_directory(model_path), \
              'saved model does not exists in %s' % model_path
          self._is_saved_model = True
          meta_graph_def = tf.saved_model.loader.load(
              self._session, [tf.saved_model.tag_constants.SERVING], model_path)
          # parse signature
          signature_def = meta_graph_def.signature_def[
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
          inputs = signature_def.inputs
          # each input_info is a tuple of input_id, name, data_type
          input_info = []
          self._is_multi_placeholder = len(inputs.items()) > 1
          if self._is_multi_placeholder:
            for gid, item in enumerate(inputs.items()):
              name, tensor = item
              logging.info('Load input binding: %s -> %s' % (name, tensor.name))
              input_name = tensor.name
              input_name, _ = input_name.split(':')
              try:
                input_id = input_name.split('_')[-1]
                input_id = int(input_id)
              except Exception:
                # support for models that are not exported by easy_rec
                # in which case, the order of inputs may not be the
                # same as they are defined, thereforce, list input
                # could not be supported, only dict input could be supported
                logging.warning(
                    'could not determine input_id from input_name: %s' %
                    input_name)
                input_id = gid
              input_info.append((input_id, name, tensor.dtype))
              self._inputs_map[name] = self._graph.get_tensor_by_name(
                  tensor.name)
          else:
            # only one input, all features concatenate together
            for name, tensor in inputs.items():
              logging.info('Load input binding: %s -> %s' % (name, tensor.name))
              input_info.append((0, name, tensor.dtype))
              self._inputs_map[name] = self._graph.get_tensor_by_name(
                  tensor.name)
          # sort inputs by input_ids so as to match the order of csv data
          input_info.sort(key=lambda t: t[0])
          self._input_names = [t[1] for t in input_info]

          outputs = signature_def.outputs
          for name, tensor in outputs.items():
            logging.info('Load output binding: %s -> %s' % (name, tensor.name))
            self._outputs_map[name] = self._graph.get_tensor_by_name(
                tensor.name)

          # get assets
          self._assets = {}
          asset_files = tf.get_collection(constants.ASSETS_KEY)
          for any_proto in asset_files:
            asset_file = meta_graph_pb2.AssetFileDef()
            any_proto.Unpack(asset_file)
            type_name = asset_file.tensor_info.name.split(':')[0]
            asset_path = os.path.join(model_path, constants.ASSETS_DIRECTORY,
                                      asset_file.filename)
            assert gfile.Exists(
                asset_path), '%s is missing in saved model' % asset_path
            self._assets[type_name] = asset_path
          logging.info(self._assets)

          # get export config
          self._export_config = {}
          # export_config_collection = tf.get_collection(fields.EVGraphKeys.export_config)
          # if len(export_config_collection) > 0:
          #  self._export_config = json.loads(export_config_collection[0])
          #  logging.info('load export config info %s' % export_config_collection[0])
        else:
          raise ValueError('currently only savedmodel is supported')

  def predict(self, input_data_dict, output_names=None):
    """Predict input data with loaded model.

    Args:
      input_data_dict: a dict containing all input data, key is the input name,
        value is the corresponding value
      output_names:  if not None, will fetch certain outputs, if set None, will
        return all the output info according to the output info in model signature

    Return:
      a dict of outputs, key is the output name, value is the corresponding value
    """
    feed_dict = {}
    for input_name, tensor in six.iteritems(self._inputs_map):
      assert input_name in input_data_dict, 'input data %s is missing' % input_name
      tensor_shape = tensor.get_shape().as_list()
      input_shape = input_data_dict[input_name].shape
      assert tensor_shape[0] is None or (tensor_shape[0] == input_shape[0]), \
          'input %s  batchsize %d is not the same as the exported batch_size %d' % \
          (input_name, input_shape[0], tensor_shape[0])
      feed_dict[tensor] = input_data_dict[input_name]
    fetch_dict = {}
    if output_names is not None:
      for output_name in output_names:
        assert output_name in self._outputs_map, \
            'invalid output name %s' % output_name
        fetch_dict[output_name] = self._outputs_map[output_name]
    else:
      fetch_dict = self._outputs_map

    with self._graph.as_default():
      with self._session.as_default():
        if self._profiling_file is None:
          return self._session.run(fetch_dict, feed_dict)
        else:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          results = self._session.run(
              fetch_dict,
              feed_dict,
              options=run_options,
              run_metadata=run_metadata)
          # Create the Timeline object, and write it to a json
          from tensorflow.python.client import timeline
          tl = timeline.Timeline(run_metadata.step_stats)
          ctf = tl.generate_chrome_trace_format()
          with gfile.GFile(self._profiling_file, 'w') as f:
            f.write(ctf)
          return results


class Predictor(PredictorInterface):

  def __init__(self,
               model_path,
               profiling_file=None,
               fg_json_path=None,
               use_latest=True):
    """Initialize a `Predictor`.

    Args:
      model_path:  saved_model directory or frozenpb file path
      profiling_file:  profiling result file, default None.
        if not None, predict function will use Timeline to profiling
        prediction time, and the result json will be saved to profiling_file
      fg_json_path: fg.json file
      use_latest: use latest saved_model.pb if multiple one exists.
    """
    self._predictor_impl = PredictorImpl(model_path, profiling_file, use_latest)
    self._inputs_map = self._predictor_impl._inputs_map
    self._outputs_map = self._predictor_impl._outputs_map
    self._profiling_file = profiling_file
    self._export_config = self._predictor_impl._export_config
    self._input_fields_info = self._predictor_impl._input_fields_info
    self._is_multi_placeholder = self._predictor_impl._is_multi_placeholder

    self._input_fields = self._predictor_impl._input_fields_list
    fg_json = self._get_fg_json(fg_json_path, model_path)
    self._all_input_names = get_input_name_from_fg_json(fg_json)
    logging.info('all_input_names: %s' % self._all_input_names)

  @property
  def input_names(self):
    """Input names of the model.

    Returns:
      a list, which conaining the name of input nodes available in model
    """
    return list(self._inputs_map.keys())

  @property
  def output_names(self):
    """Output names of the model.

    Returns:
      a list, which conaining the name of outputs nodes available in model
    """
    return list(self._outputs_map.keys())

  def _get_defaults(self, col_name, col_type='string'):
    if col_name in self._input_fields_info:
      col_type, default_val = self._input_fields_info[col_name]
      default_val = get_type_defaults(col_type, default_val)
      logging.info('col_name: %s, default_val: %s' % (col_name, default_val))
    else:
      defaults = {'string': '', 'double': 0.0, 'bigint': 0}
      assert col_type in defaults, 'invalid col_type: %s, col_type: %s' % (
          col_name, col_type)
      default_val = defaults[col_type]
      logging.info(
          'col_name: %s, default_val: %s.[not defined in saved_model_dir/assets/pipeline.config]'
          % (col_name, default_val))
    return default_val

  def _parse_line(self, line):
    pass

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    pass

  def _get_writer(self, output_path, slice_id):
    pass

  def _get_reserved_cols(self, reserved_cols):
    pass

  @property
  def out_of_range_exception(self):
    return None

  def _write_lines(self, table_writer, outputs):
    pass

  def load_to_table(self, output_path, slice_num, slice_id):
    pass

  def _get_fg_json(self, fg_json_path, model_path):
    if fg_json_path and gfile.Exists(fg_json_path):
      logging.info('load fg_json_path: ', fg_json_path)
      with tf.gfile.GFile(fg_json_path, 'r') as fin:
        fg_json = json.loads(fin.read())
    else:
      fg_json_path = search_fg_json(model_path)
      if fg_json_path:
        with tf.gfile.GFile(fg_json_path, 'r') as fin:
          fg_json = json.loads(fin.read())
      else:
        fg_json = {}
    return fg_json

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    pass

  def predict_impl(
      self,
      input_path,
      output_path,
      reserved_cols='',
      output_cols=None,
      batch_size=1024,
      slice_id=0,
      slice_num=1,
  ):
    """Predict table input with loaded model.

    Args:
      input_path: table/file_path to read
      output_path: table/file_path to write
      reserved_cols: columns to be copy to output_table, comma separated, such as "a,b"
      output_cols: output columns, comma separated, such as "y float, embedding string",
                the output names[y, embedding] must be in saved_model output_names
      batch_size: predict batch size
      slice_id: when multiple workers write the same table, each worker should
                be assigned different slice_id, which is usually slice_id
      slice_num: table slice number
    """
    if output_cols is None or output_cols == 'ALL_COLUMNS':
      self._output_cols = sorted(self._predictor_impl.output_names)
      logging.info('predict output cols: %s' % self._output_cols)
    else:
      # specified as score float,embedding string
      tmp_cols = []
      for x in output_cols.split(','):
        if x.strip() == '':
          continue
        tmp_keys = x.split(' ')
        tmp_cols.append(tmp_keys[0].strip())
      self._output_cols = tmp_cols

    with tf.Graph().as_default(), tf.Session() as sess:
      num_parallel_calls = 8
      dataset = self._get_dataset(input_path, num_parallel_calls, batch_size,
                                  slice_num, slice_id)
      dataset = dataset.map(
          self._parse_line, num_parallel_calls=num_parallel_calls)
      iterator = dataset.make_one_shot_iterator()
      all_dict = iterator.get_next()
      self._reserved_cols = self._get_reserved_cols(reserved_cols)
      input_names = self._predictor_impl.input_names
      table_writer = self._get_writer(output_path, slice_id)

      def _parse_value(all_vals):
        if self._is_multi_placeholder:
          if SINGLE_PLACEHOLDER_FEATURE_KEY in all_vals:
            feature_vals = all_vals[SINGLE_PLACEHOLDER_FEATURE_KEY]
            split_index = []
            split_vals = {}
            fg_input_size = len(feature_vals[0].decode('utf-8').split('\002'))
            if fg_input_size == len(input_names):
              for i, k in enumerate(input_names):
                split_index.append(k)
                split_vals[k] = []
            else:
              assert self._all_input_names, 'must set fg_json_path when use fg input'
              assert fg_input_size == len(self._all_input_names), (
                  'The number of features defined in fg_json != the size of fg input. '
                  'The number of features defined in fg_json is: %d; The size of fg input is: %d'
                  % (len(self._all_input_names), fg_input_size))
              for i, k in enumerate(self._all_input_names):
                split_index.append(k)
                split_vals[k] = []
            for record in feature_vals:
              split_records = record.decode('utf-8').split('\002')
              for i, r in enumerate(split_records):
                split_vals[split_index[i]].append(r)
            return {k: np.array(split_vals[k]) for k in input_names}
        return {k: all_vals[k] for k in input_names}

      progress = 0
      sum_t0, sum_t1, sum_t2 = 0, 0, 0

      while True:
        try:
          ts0 = time.time()
          all_vals = sess.run(all_dict)

          ts1 = time.time()
          input_vals = _parse_value(all_vals)
          outputs = self._predictor_impl.predict(input_vals, self._output_cols)
          for x in self._output_cols:
            if outputs[x].dtype == np.object:
              outputs[x] = [val.decode('utf-8') for val in outputs[x]]
            elif len(outputs[x].shape) > 1:
              outputs[x] = [
                  json.dumps(val, cls=numpy_utils.NumpyEncoder)
                  for val in outputs[x]
              ]
          for k in self._reserved_cols:
            if all_vals[k].dtype == np.object:
              all_vals[k] = [val.decode('utf-8') for val in all_vals[k]]

          ts2 = time.time()
          reserve_vals = self._get_reserve_vals(self._reserved_cols,
                                                self._output_cols, all_vals,
                                                outputs)
          outputs = [x for x in zip(*reserve_vals)]
          logging.info('predict size: %s' % len(outputs))
          self._write_lines(table_writer, outputs)

          ts3 = time.time()
          progress += 1
          sum_t0 += (ts1 - ts0)
          sum_t1 += (ts2 - ts1)
          sum_t2 += (ts3 - ts2)
        except self.out_of_range_exception:
          break
        if progress % 100 == 0:
          logging.info('progress: batch_num=%d sample_num=%d' %
                       (progress, progress * batch_size))
          logging.info('time_stats: read: %.2f predict: %.2f write: %.2f' %
                       (sum_t0, sum_t1, sum_t2))
      logging.info('Final_time_stats: read: %.2f predict: %.2f write: %.2f' %
                   (sum_t0, sum_t1, sum_t2))
      table_writer.close()
      self.load_to_table(output_path, slice_num, slice_id)
      logging.info('Predict %s done.' % input_path)

  def predict(self, input_data_dict_list, output_names=None, batch_size=1):
    """Predict input data with loaded model.

    Args:
      input_data_dict_list: list of dict
      output_names:  if not None, will fetch certain outputs, if set None, will
      batch_size: batch_size used to predict, -1 indicates to use the real batch_size

    Return:
      a list of dict, each dict contain a key-value pair for output_name, output_value
    """
    num_example = len(input_data_dict_list)
    assert num_example > 0, 'input data should not be an empty list'
    assert isinstance(input_data_dict_list[0], dict) or \
           isinstance(input_data_dict_list[0], list) or \
           isinstance(input_data_dict_list[0], str), 'input is not a list or dict or str'
    if batch_size > 0:
      num_batches = int(math.ceil(float(num_example) / batch_size))
    else:
      num_batches = 1
      batch_size = len(input_data_dict_list)

    outputs_list = []
    for batch_idx in range(num_batches):
      batch_data_list = input_data_dict_list[batch_idx *
                                             batch_size:(batch_idx + 1) *
                                             batch_size]
      feed_dict = self.batch(batch_data_list)
      outputs = self._predictor_impl.predict(feed_dict, output_names)
      for idx in range(len(batch_data_list)):
        single_result = {}
        for key, batch_value in six.iteritems(outputs):
          single_result[key] = batch_value[idx]
        outputs_list.append(single_result)
    return outputs_list

  def batch(self, data_list):
    """Batching the data."""
    batch_input = {key: [] for key in self._predictor_impl.input_names}
    for data in data_list:
      if isinstance(data, dict):
        for key in data:
          batch_input[key].append(data[key])
      elif isinstance(data, list):
        assert len(self._predictor_impl.input_names) == len(data), \
            'input fields number incorrect, should be %d, but %d' \
            % (len(self._predictor_impl.input_names), len(data))
        for key, v in zip(self._predictor_impl.input_names, data):
          if key != '':
            batch_input[key].append(v)
      elif isinstance(data, str):
        batch_input[self._predictor_impl.input_names[0]].append(data)
    for key in batch_input:
      batch_input[key] = np.array(batch_input[key])
    return batch_input


class CSVPredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               ds_vector_recall=False,
               fg_json_path=None,
               profiling_file=None,
               selected_cols=None,
               output_sep=chr(1)):
    super(CSVPredictor, self).__init__(model_path, profiling_file, fg_json_path)
    self._output_sep = output_sep
    self._ds_vector_recall = ds_vector_recall
    input_type = DatasetConfig.InputType.Name(data_config.input_type).lower()
    self._with_header = data_config.with_header

    if 'rtp' in input_type:
      self._is_rtp = True
      self._input_sep = data_config.rtp_separator
    else:
      self._is_rtp = False
      self._input_sep = data_config.separator

    if selected_cols and not ds_vector_recall:
      self._selected_cols = [int(x) for x in selected_cols.split(',')]
    elif ds_vector_recall:
      self._selected_cols = selected_cols.split(',')
    else:
      self._selected_cols = None

  def _get_reserved_cols(self, reserved_cols):
    if reserved_cols == 'ALL_COLUMNS':
      if self._is_rtp:
        if self._with_header:
          reserved_cols = self._all_fields
        else:
          idx = 0
          reserved_cols = []
          for x in range(len(self._record_defaults) - 1):
            if not self._selected_cols or x in self._selected_cols[:-1]:
              reserved_cols.append(self._input_fields[idx])
              idx += 1
            else:
              reserved_cols.append('no_used_%d' % x)
          reserved_cols.append(SINGLE_PLACEHOLDER_FEATURE_KEY)
      else:
        reserved_cols = self._all_fields
    else:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, line):
    check_list = [
        tf.py_func(
            check_split, [line, self._input_sep,
                          len(self._record_defaults)],
            Tout=tf.bool)
    ]
    with tf.control_dependencies(check_list):
      fields = tf.decode_csv(
          line,
          field_delim=self._input_sep,
          record_defaults=self._record_defaults,
          name='decode_csv')
    if self._is_rtp:
      if self._with_header:
        inputs = dict(zip(self._all_fields, fields))
      else:
        inputs = {}
        idx = 0
        for x in range(len(self._record_defaults) - 1):
          if not self._selected_cols or x in self._selected_cols[:-1]:
            inputs[self._input_fields[idx]] = fields[x]
            idx += 1
          else:
            inputs['no_used_%d' % x] = fields[x]
        inputs[SINGLE_PLACEHOLDER_FEATURE_KEY] = fields[-1]
    else:
      inputs = {self._all_fields[x]: fields[x] for x in range(len(fields))}
    return inputs

  def _get_num_cols(self, file_paths):
    # try to figure out number of fields from one file
    num_cols = -1
    with tf.gfile.GFile(file_paths[0], 'r') as fin:
      num_lines = 0
      for line_str in fin:
        line_tok = line_str.strip().split(self._input_sep)
        if num_cols != -1:
          assert num_cols == len(line_tok), (
              'num selected cols is %d, not equal to %d, current line is: %s, please check input_sep and data.'
              % (num_cols, len(line_tok), line_str))
        num_cols = len(line_tok)
        num_lines += 1
        if num_lines > 10:
          break
    logging.info('num selected cols = %d' % num_cols)
    return num_cols

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    file_paths = []
    for path in input_path.split(','):
      for x in gfile.Glob(path):
        if not x.endswith('_SUCCESS'):
          file_paths.append(x)
    assert len(file_paths) > 0, 'match no files with %s' % input_path

    if self._with_header:
      with tf.gfile.GFile(file_paths[0], 'r') as fin:
        for line_str in fin:
          line_str = line_str.strip()
          self._field_names = line_str.split(self._input_sep)
          break
      print('field_names: %s' % ','.join(self._field_names))
      self._all_fields = self._field_names
    elif self._ds_vector_recall:
      self._all_fields = self._selected_cols
    else:
      self._all_fields = self._input_fields
    if self._is_rtp:
      num_cols = self._get_num_cols(file_paths)
      self._record_defaults = ['' for _ in range(num_cols)]
      if not self._selected_cols:
        self._selected_cols = list(range(num_cols))
      for col_idx in self._selected_cols[:-1]:
        col_name = self._input_fields[col_idx]
        default_val = self._get_defaults(col_name)
        self._record_defaults[col_idx] = default_val
    else:
      self._record_defaults = [
          self._get_defaults(col_name) for col_name in self._all_fields
      ]

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    parallel_num = min(num_parallel_calls, len(file_paths))
    dataset = dataset.interleave(
        lambda x: tf.data.TextLineDataset(x).skip(int(self._with_header)),
        cycle_length=parallel_num,
        num_parallel_calls=parallel_num)
    dataset = dataset.shard(slice_num, slice_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  def _get_writer(self, output_path, slice_id):
    if not gfile.Exists(output_path):
      gfile.MakeDirs(output_path)
    res_path = os.path.join(output_path, 'part-%d.csv' % slice_id)
    table_writer = gfile.GFile(res_path, 'w')
    table_writer.write(
        self._output_sep.join(self._output_cols + self._reserved_cols) + '\n')
    return table_writer

  def _write_lines(self, table_writer, outputs):
    outputs = '\n'.join(
        [self._output_sep.join([str(i) for i in output]) for output in outputs])
    table_writer.write(outputs + '\n')

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [outputs[x] for x in output_cols] + \
                   [all_vals[k] for k in reserved_cols]
    return reserve_vals

  @property
  def out_of_range_exception(self):
    return (tf.errors.OutOfRangeError)


class ODPSPredictor(Predictor):

  def __init__(self,
               model_path,
               fg_json_path=None,
               profiling_file=None,
               all_cols='',
               all_col_types=''):
    super(ODPSPredictor, self).__init__(model_path, profiling_file,
                                        fg_json_path)
    self._all_cols = [x.strip() for x in all_cols.split(',') if x != '']
    self._all_col_types = [
        x.strip() for x in all_col_types.split(',') if x != ''
    ]
    self._record_defaults = [
        self._get_defaults(col_name, col_type)
        for col_name, col_type in zip(self._all_cols, self._all_col_types)
    ]

  def _get_reserved_cols(self, reserved_cols):
    reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, *fields):
    fields = list(fields)
    field_dict = {self._all_cols[i]: fields[i] for i in range(len(fields))}
    return field_dict

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    input_list = input_path.split(',')
    dataset = tf.data.TableRecordDataset(
        input_list,
        record_defaults=self._record_defaults,
        slice_id=slice_id,
        slice_count=slice_num,
        selected_cols=','.join(self._all_cols))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  def _get_writer(self, output_path, slice_id):
    import common_io
    table_writer = common_io.table.TableWriter(output_path, slice_id=slice_id)
    return table_writer

  def _write_lines(self, table_writer, outputs):
    assert len(outputs) > 0
    indices = list(range(0, len(outputs[0])))
    table_writer.write(outputs, indices, allow_type_cast=False)

  @property
  def out_of_range_exception(self):
    return (tf.python_io.OutOfRangeException, tf.errors.OutOfRangeError)

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [all_vals[k] for k in reserved_cols] + \
                   [outputs[x] for x in output_cols]
    return reserve_vals


class HivePredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               hive_config,
               fg_json_path=None,
               profiling_file=None,
               output_sep=chr(1),
               all_cols=None,
               all_col_types=None):
    super(HivePredictor, self).__init__(model_path, profiling_file,
                                        fg_json_path)

    self._data_config = data_config
    self._hive_config = hive_config
    self._output_sep = output_sep
    input_type = DatasetConfig.InputType.Name(data_config.input_type).lower()
    if 'rtp' in input_type:
      self._is_rtp = True
    else:
      self._is_rtp = False
    self._all_cols = [x.strip() for x in all_cols if x != '']
    self._all_col_types = [x.strip() for x in all_col_types if x != '']
    self._record_defaults = [
        self._get_defaults(col_name, col_type)
        for col_name, col_type in zip(self._all_cols, self._all_col_types)
    ]

  def _get_reserved_cols(self, reserved_cols):
    if reserved_cols == 'ALL_COLUMNS':
      reserved_cols = self._all_cols
    else:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, line):
    field_delim = self._data_config.rtp_separator if self._is_rtp else self._data_config.separator
    fields = tf.decode_csv(
        line,
        field_delim=field_delim,
        record_defaults=self._record_defaults,
        name='decode_csv')
    inputs = {self._all_cols[x]: fields[x] for x in range(len(fields))}
    return inputs

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    self._hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    self._input_hdfs_path = self._hive_util.get_table_location(input_path)
    file_paths = tf.gfile.Glob(os.path.join(self._input_hdfs_path, '*'))
    assert len(file_paths) > 0, 'match no files with %s' % input_path

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    parallel_num = min(num_parallel_calls, len(file_paths))
    dataset = dataset.interleave(
        tf.data.TextLineDataset,
        cycle_length=parallel_num,
        num_parallel_calls=parallel_num)
    dataset = dataset.shard(slice_num, slice_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  def get_table_info(self, output_path):
    partition_name, partition_val = None, None
    if len(output_path.split('/')) == 2:
      table_name, partition = output_path.split('/')
      partition_name, partition_val = partition.split('=')
    else:
      table_name = output_path
    return table_name, partition_name, partition_val

  def _get_writer(self, output_path, slice_id):
    table_name, partition_name, partition_val = self.get_table_info(output_path)
    is_exist = self._hive_util.is_table_or_partition_exist(
        table_name, partition_name, partition_val)
    assert not is_exist, '%s is already exists. Please drop it.' % output_path

    output_path = output_path.replace('.', '/')
    self._hdfs_path = 'hdfs://%s:9000/user/easy_rec/%s_tmp' % (
        self._hive_config.host, output_path)
    if not gfile.Exists(self._hdfs_path):
      gfile.MakeDirs(self._hdfs_path)
    res_path = os.path.join(self._hdfs_path, 'part-%d.csv' % slice_id)
    table_writer = gfile.GFile(res_path, 'w')
    return table_writer

  def _write_lines(self, table_writer, outputs):
    outputs = '\n'.join(
        [self._output_sep.join([str(i) for i in output]) for output in outputs])
    table_writer.write(outputs + '\n')

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [outputs[x] for x in output_cols] + \
                   [all_vals[k] for k in reserved_cols]
    return reserve_vals

  def load_to_table(self, output_path, slice_num, slice_id):
    res_path = os.path.join(self._hdfs_path, 'SUCCESS-%s' % slice_id)
    success_writer = gfile.GFile(res_path, 'w')
    success_writer.write('')
    success_writer.close()

    if slice_id != 0:
      return

    for id in range(slice_num):
      res_path = os.path.join(self._hdfs_path, 'SUCCESS-%s' % id)
      while not gfile.Exists(res_path):
        time.sleep(10)

    table_name, partition_name, partition_val = self.get_table_info(output_path)
    schema = ''
    for output_col_name in self._output_cols:
      tf_type = self._predictor_impl._outputs_map[output_col_name].dtype
      col_type = tf_utils.get_col_type(tf_type)
      schema += output_col_name + ' ' + col_type + ','

    for output_col_name in self._reserved_cols:
      assert output_col_name in self._all_cols, 'Column: %s not exists.' % output_col_name
      idx = self._all_cols.index(output_col_name)
      output_col_types = self._all_col_types[idx]
      schema += output_col_name + ' ' + output_col_types + ','
    schema = schema.rstrip(',')

    if partition_name and partition_val:
      sql = 'create table if not exists %s (%s) PARTITIONED BY (%s string)' % \
            (table_name, schema, partition_name)
      self._hive_util.run_sql(sql)
      sql = "LOAD DATA INPATH '%s/*' INTO TABLE %s PARTITION (%s=%s)" % \
            (self._hdfs_path, table_name, partition_name, partition_val)
      self._hive_util.run_sql(sql)
    else:
      sql = 'create table if not exists %s (%s)' % \
            (table_name, schema)
      self._hive_util.run_sql(sql)
      sql = "LOAD DATA INPATH '%s/*' INTO TABLE %s" % \
            (self._hdfs_path, table_name)
      self._hive_util.run_sql(sql)

  @property
  def out_of_range_exception(self):
    return (tf.errors.OutOfRangeError)


class HiveParquetPredictor(Predictor):

  def __init__(self,
               model_path,
               data_config,
               hive_config,
               fg_json_path=None,
               profiling_file=None,
               output_sep=chr(1),
               all_cols=None,
               all_col_types=None):
    super(HiveParquetPredictor, self).__init__(model_path, profiling_file,
                                               fg_json_path)

    self._data_config = data_config
    self._hive_config = hive_config
    self._output_sep = output_sep
    input_type = DatasetConfig.InputType.Name(data_config.input_type).lower()
    if 'rtp' in input_type:
      self._is_rtp = True
    else:
      self._is_rtp = False
    self._all_cols = [x.strip() for x in all_cols if x != '']
    self._all_col_types = [x.strip() for x in all_col_types if x != '']
    self._record_defaults = [
        self._get_defaults(col_name, col_type)
        for col_name, col_type in zip(self._all_cols, self._all_col_types)
    ]

  def _get_reserved_cols(self, reserved_cols):
    if reserved_cols == 'ALL_COLUMNS':
      reserved_cols = self._all_cols
    else:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, *fields):
    fields = list(fields)
    field_dict = {self._all_cols[i]: fields[i] for i in range(len(fields))}
    return field_dict

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    self._hive_util = HiveUtils(
        data_config=self._data_config, hive_config=self._hive_config)
    hdfs_path = self._hive_util.get_table_location(input_path)
    self._input_hdfs_path = tf.gfile.Glob(os.path.join(hdfs_path, '*'))
    assert len(self._input_hdfs_path) > 0, 'match no files with %s' % input_path

    list_type = []
    input_field_type_map = {
        x.input_name: x.input_type for x in self._data_config.input_fields
    }
    type_2_tftype = {
        'string': tf.string,
        'double': tf.double,
        'float': tf.float32,
        'bigint': tf.int32,
        'boolean': tf.bool
    }
    for col_name, col_type in zip(self._all_cols, self._all_col_types):
      if col_name in input_field_type_map:
        list_type.append(get_tf_type(input_field_type_map[col_name]))
      else:
        list_type.append(type_2_tftype[col_type.lower()])
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    def parquet_read():
      for input_path in self._input_hdfs_path:
        if input_path.endswith('SUCCESS'):
          continue
        df = pd.read_parquet(input_path, engine='pyarrow')

        df.replace('', np.nan, inplace=True)
        df.replace('NULL', np.nan, inplace=True)
        total_records_num = len(df)

        for k, v in zip(self._all_cols, self._record_defaults):
          df[k].fillna(v, inplace=True)

        for start_idx in range(0, total_records_num, batch_size):
          end_idx = min(total_records_num, start_idx + batch_size)
          batch_data = df[start_idx:end_idx]
          inputs = []
          for k in self._all_cols:
            inputs.append(batch_data[k].to_numpy())
          yield tuple(inputs)

    dataset = tf.data.Dataset.from_generator(
        parquet_read, output_types=list_type, output_shapes=list_shapes)
    dataset = dataset.shard(slice_num, slice_id)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  def get_table_info(self, output_path):
    partition_name, partition_val = None, None
    if len(output_path.split('/')) == 2:
      table_name, partition = output_path.split('/')
      partition_name, partition_val = partition.split('=')
    else:
      table_name = output_path
    return table_name, partition_name, partition_val

  def _get_writer(self, output_path, slice_id):
    table_name, partition_name, partition_val = self.get_table_info(output_path)
    is_exist = self._hive_util.is_table_or_partition_exist(
        table_name, partition_name, partition_val)
    assert not is_exist, '%s is already exists. Please drop it.' % output_path

    output_path = output_path.replace('.', '/')
    self._hdfs_path = 'hdfs://%s:9000/user/easy_rec/%s_tmp' % (
        self._hive_config.host, output_path)
    if not gfile.Exists(self._hdfs_path):
      gfile.MakeDirs(self._hdfs_path)
    res_path = os.path.join(self._hdfs_path, 'part-%d.csv' % slice_id)
    table_writer = gfile.GFile(res_path, 'w')
    return table_writer

  def _write_lines(self, table_writer, outputs):
    outputs = '\n'.join(
        [self._output_sep.join([str(i) for i in output]) for output in outputs])
    table_writer.write(outputs + '\n')

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [outputs[x] for x in output_cols] + \
                   [all_vals[k] for k in reserved_cols]
    return reserve_vals

  def load_to_table(self, output_path, slice_num, slice_id):
    res_path = os.path.join(self._hdfs_path, 'SUCCESS-%s' % slice_id)
    success_writer = gfile.GFile(res_path, 'w')
    success_writer.write('')
    success_writer.close()

    if slice_id != 0:
      return

    for id in range(slice_num):
      res_path = os.path.join(self._hdfs_path, 'SUCCESS-%s' % id)
      while not gfile.Exists(res_path):
        time.sleep(10)

    table_name, partition_name, partition_val = self.get_table_info(output_path)
    schema = ''
    for output_col_name in self._output_cols:
      tf_type = self._predictor_impl._outputs_map[output_col_name].dtype
      col_type = tf_utils.get_col_type(tf_type)
      schema += output_col_name + ' ' + col_type + ','

    for output_col_name in self._reserved_cols:
      assert output_col_name in self._all_cols, 'Column: %s not exists.' % output_col_name
      idx = self._all_cols.index(output_col_name)
      output_col_types = self._all_col_types[idx]
      schema += output_col_name + ' ' + output_col_types + ','
    schema = schema.rstrip(',')

    if partition_name and partition_val:
      sql = 'create table if not exists %s (%s) PARTITIONED BY (%s string)' % \
            (table_name, schema, partition_name)
      self._hive_util.run_sql(sql)
      sql = "LOAD DATA INPATH '%s/*' INTO TABLE %s PARTITION (%s=%s)" % \
            (self._hdfs_path, table_name, partition_name, partition_val)
      self._hive_util.run_sql(sql)
    else:
      sql = 'create table if not exists %s (%s)' % \
            (table_name, schema)
      self._hive_util.run_sql(sql)
      sql = "LOAD DATA INPATH '%s/*' INTO TABLE %s" % \
            (self._hdfs_path, table_name)
      self._hive_util.run_sql(sql)

  @property
  def out_of_range_exception(self):
    return (tf.errors.OutOfRangeError)
