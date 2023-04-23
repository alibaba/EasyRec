import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from easy_rec.python.utils.load_class import get_register_class_meta
from easy_rec.python.utils.config_util import get_configs_from_pipeline_file
from easy_rec.python.utils.input_utils import get_type_defaults
from easy_rec.python.tools.explainer.methods import DeepExplain
# from easy_rec.python.tools.explainer.deep_shap import DeepShap
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
import abc
import collections
import numpy as np
import logging
import six
import time
from six import moves
import os

_EXPLAINER_CLASS_MAP = {}
_register_abc_meta = get_register_class_meta(
  _EXPLAINER_CLASS_MAP, have_abstract_class=True)


class Explainer(six.with_metaclass(_register_abc_meta, object)):
  version = 1

  def __init__(self, deep_explain, model_path, method_name):
    """Base class for explainer.

    Args:
      deep_explain: a deep explain context manager
      model_path:  saved_model directory or frozen pb file path
      method_name: explain method name
    """
    self.deep_explain = deep_explain
    self.method = method_name
    self._inputs_map = collections.OrderedDict()
    self._outputs_map = collections.OrderedDict()
    self._model_path = model_path
    self._explainer = None
    self._effective_fields = None
    self._build_model()

  def _build_model(self):
    model_path = self._model_path
    logging.info('loading model from %s' % model_path)
    if gfile.IsDirectory(model_path):
      assert tf.saved_model.loader.maybe_saved_model_directory(model_path), \
        'saved model does not exists in %s' % model_path
    else:
      raise ValueError('currently only savedmodel is supported, path:' + model_path)

    input_fields = _get_input_fields_from_pipeline_config(model_path)
    self._input_fields_info, self._input_fields = input_fields

    de = self.deep_explain
    meta_graph_def = tf.saved_model.loader.load(
      de.session, [tf.saved_model.tag_constants.SERVING], model_path)
    # parse signature
    signature_def = meta_graph_def.signature_def[
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    inputs = signature_def.inputs
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
          # same as they are defined, therefore, list input
          # could not be supported, only dict input could be supported
          logging.warning(
            'could not determine input_id from input_name: %s' % input_name)
          input_id = gid
        input_info.append((input_id, name, tensor.dtype))
        self._inputs_map[name] = de.graph.get_tensor_by_name(tensor.name)
    else:
      # only one input, all features concatenate together
      for name, tensor in inputs.items():
        logging.info('Load input binding: %s -> %s' % (name, tensor.name))
        input_info.append((0, name, tensor.dtype))
        self._inputs_map[name] = de.graph.get_tensor_by_name(tensor.name)

    # sort inputs by input_ids so as to match the order of csv data
    input_info.sort(key=lambda t: t[0])
    self._input_names = [t[1] for t in input_info]

    outputs = signature_def.outputs
    for name, tensor in outputs.items():
      logging.info('Load output binding: %s -> %s' % (name, tensor.name))
      self._outputs_map[name] = de.graph.get_tensor_by_name(tensor.name)

    # get assets
    # self._assets = {}
    # asset_files = tf.get_collection(constants.ASSETS_KEY)
    # for any_proto in asset_files:
    #   asset_file = meta_graph_pb2.AssetFileDef()
    #   any_proto.Unpack(asset_file)
    #   type_name = asset_file.tensor_info.name.split(':')[0]
    #   asset_path = os.path.join(model_path, constants.ASSETS_DIRECTORY,
    #                             asset_file.filename)
    #   assert gfile.Exists(
    #     asset_path), '%s is missing in saved model' % asset_path
    #   self._assets[type_name] = asset_path
    # logging.info(self._assets)

  def default_values(self):
    input_fields = self._input_fields if self._effective_fields is None else self._effective_fields
    n = len(input_fields)
    m = len(self._input_names)
    assert m == n, 'the number input columns is not expected, %d given, %d expected\n' \
                   'model inputs: %s\ninput fields: %s' % (n, m, ','.join(self._input_names), ','.join(input_fields))

    default_value = []
    for i, (field, name) in enumerate(zip(input_fields, self._input_names)):
      assert field == name, "input field `%d` has different names: <%s, %s>" % (i, field, name)
      value = self._get_defaults(field)
      # default_value.append(np.array([value]))  # for deep_shap
      default_value.append(np.array(value))  # for deep_shap
    return default_value

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

  def str_to_number(self, values):
    assert len(values) == len(self._input_fields), "value count %d is not equal to the number of input fields %d" % (
      len(values), len(self._input_fields)
    )
    result = []
    for i, name in enumerate(self._input_names):
      assert name in self._input_fields_info, "input `%s` not in pipeline config" % name
      idx = self._input_fields.index(name)
      input_type, default_val = self._input_fields_info[name]
      if input_type in {DatasetConfig.INT32, DatasetConfig.INT64}:
        tmp_field = int(values[idx])
      elif input_type in [DatasetConfig.FLOAT, DatasetConfig.DOUBLE]:
        tmp_field = float(values[idx])
      elif input_type in [DatasetConfig.BOOL]:
        tmp_field = values[idx].lower() in ['true', '1', 't', 'y', 'yes']
      elif input_type in [DatasetConfig.STRING]:
        tmp_field = values[idx]
      else:
        assert False, 'invalid types: %s' % str(input_type)
      result.append(tmp_field)
    return result

  def get_explainer(self, output_cols=None):
    if output_cols is None or output_cols == 'ALL_COLUMNS':
      self._output_cols = sorted(self.output_names)
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
    if len(self._output_cols) > 1:
      logging.warning('Only one output can be supported currently, use the first one: %s', self._output_cols[0])

    output_name = self._output_cols[0]
    assert output_name in self.output_names, 'invalid output name `%s` not in model outputs `%s`' % (
      output_name, ','.join(self.output_names))
    if output_name is None:
      output = self._outputs_map.values()[0]
    elif type(output_name) in {str, unicode}:
      output = self._outputs_map[output_name]
    else:
      raise Exception('unsupported type of output_name: ' + str(type(output_name)))

    def_vals = self.default_values()
    # print('default values (%d):' % len(def_vals), def_vals)
    inputs = [self._inputs_map[name] for name in self._input_names]
    # e = DeepShap(inputs, output, def_vals, session=self._session)
    # self._explainer = e
    e = self.deep_explain.get_explainer(self.method, output, inputs, baseline=def_vals)
    return e

  @property
  def input_names(self):
    """Input names of the model.

    Returns:
      a list, which conaining the name of input nodes available in model
    """
    return self._input_names

  @property
  def output_names(self):
    """Output names of the model.

    Returns:
      a list, which containing the name of outputs nodes available in model
    """
    return list(self._outputs_map.keys())

  @abc.abstractmethod
  def feature_importance(self,
                         input_path,
                         output_path,
                         reserved_cols='',
                         output_cols=None,
                         batch_size=1024,
                         slice_id=0,
                         slice_num=1):
    pass

  # def create_output_table(self, reserved_cols=''):
  #   reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
  #   outputs = self.input_names
  #   reserved_cols = filter(lambda r: r not in outputs, reserved_cols)
  #   output_cols = reserved_cols + outputs
  #   sql = 'create table output_table '
  #   return sql


class OdpsExplainer(Explainer):
  def feature_importance(self,
                         input_path,
                         output_path,
                         reserved_cols='',
                         output_cols=None,
                         batch_size=1024,
                         slice_id=0,
                         slice_num=1):
    input_cols = self.input_names
    input_dim = len(input_cols)
    if reserved_cols:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x.strip() not in input_cols]
      input_cols.extend(reserved_cols)
    selected_cols = ','.join(input_cols)
    print("selected_cols: " + selected_cols)

    explainer = self.get_explainer(output_cols)
    print("reference value:", explainer.expected_value)

    import common_io
    reader = common_io.table.TableReader(input_path, selected_cols=selected_cols,
                                         slice_id=slice_id, slice_count=slice_num)

    reserved_cols_idx = []
    if reserved_cols:
      reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
      schema = reader.get_schema()
      columns = [str(x[0]) for x in schema]
      reserved_cols_idx = [columns.index(x) for x in reserved_cols]
      print(reserved_cols_idx)

    sum_t0, sum_t1, sum_t2 = 0, 0, 0
    writer = common_io.table.TableWriter(output_path, slice_id=slice_id)
    total_records_num = reader.get_row_count()
    for i in moves.range(0, total_records_num, batch_size):
      t0 = time.time()
      records = reader.read(batch_size, allow_smaller_final_batch=True)
      t1 = time.time()
      records = np.array(records)
      inputs = list(records[:, :input_dim].T)
      sv = explainer.shap_values(inputs, check_additivity=False)
      outputs = [records[:, i] for i in reserved_cols_idx]
      if outputs:
        outputs.extend(sv[0])
      else:
        outputs = sv[0]
      indices = range(len(outputs))
      t2 = time.time()
      writer.write(np.array(outputs).T, indices, allow_type_cast=True)
      t3 = time.time()
      sum_t0 += (t1 - t0)
      sum_t1 += (t2 - t1)
      sum_t2 += (t3 - t2)
      if i % 100 == 0:
        logging.info('progress: batch_num=%d sample_num=%d' %
                     (i + 1, (i + 1) * batch_size))
        logging.info('time_stats: read: %.2f predict: %.2f write: %.2f' %
                     (sum_t0, sum_t1, sum_t2))
      logging.info('Final_time_stats: read: %.2f predict: %.2f write: %.2f' %
                   (sum_t0, sum_t1, sum_t2))
    writer.close()
    reader.close()
    logging.info('Explain %s done.' % input_path)


class OdpsRtpExplainer(Explainer):
  def __init__(self, deep_explain, model_path, method_name):
    super(OdpsRtpExplainer, self).__init__(deep_explain, model_path, method_name)
    pipeline_path = os.path.join(model_path, 'assets/pipeline.config')
    if not gfile.Exists(pipeline_path):
      logging.warning(
        '%s not exists, default values maybe inconsistent with the values used in training.'
        % pipeline_path)
      return
    pipeline_config = get_configs_from_pipeline_file(pipeline_path)
    self._fg_separator = pipeline_config.data_config.separator

    if pipeline_config.export_config.filter_inputs:
      if len(pipeline_config.feature_configs) > 0:
        feature_configs = pipeline_config.feature_configs
      elif pipeline_config.feature_config and len(
          pipeline_config.feature_config.features) > 0:
        feature_configs = pipeline_config.feature_config.features
      else:
        assert False, 'One of feature_configs and feature_config.features must be configured.'

      self._effective_fields = []
      for fc in feature_configs:
        for input_name in fc.input_names:
          assert input_name in self._input_fields, 'invalid input_name in %s' % str(fc)
          if input_name not in self._effective_fields:
            self._effective_fields.append(input_name)
      self._effective_fids = [
        self._input_fields.index(x) for x in self._effective_fields
      ]
      # sort fids from small to large
      self._effective_fids = list(set(self._effective_fids))
      self._effective_fields = [
        self._input_fields[x] for x in self._effective_fids
      ]
      logging.info(
        "raw input fields: %d, effective fields: %d" % (len(self._input_fields), len(self._effective_fields)))

  def feature_importance(self,
                         input_path,
                         output_path,
                         reserved_cols='',
                         output_cols=None,
                         batch_size=1024,
                         slice_id=0,
                         slice_num=1):
    input_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    reserved_dim = len(input_cols)
    if 'features' not in input_cols:
      input_cols.append('features')
    selected_cols = ','.join(input_cols)
    print("selected_cols: " + selected_cols)

    explainer = self.get_explainer(output_cols)
    print("reference value:", explainer.expected_value)

    import common_io
    reader = common_io.table.TableReader(input_path, selected_cols=selected_cols,
                                         slice_id=slice_id, slice_count=slice_num)

    sum_t0, sum_t1, sum_t2 = 0, 0, 0
    writer = common_io.table.TableWriter(output_path, slice_id=slice_id)
    total_records_num = reader.get_row_count()
    for i in moves.range(0, total_records_num, batch_size):
      t0 = time.time()
      records = reader.read(batch_size, allow_smaller_final_batch=True)
      t1 = time.time()
      inputs = []
      reserved = []
      for j in range(len(records)):
        if reserved_dim > 0:
          reserved.append(records[j][:reserved_dim])
        inputs.append(self.str_to_number(records[j][-1].decode('utf-8').split(self._fg_separator)))
      inputs = list(np.array(inputs).T)
      print("inputs:", inputs)
      # sv = explainer.shap_values(inputs, check_additivity=False)
      ret = explainer.run(inputs, batch_size=len(records))
      ret = np.array(ret)
      if reserved_dim > 0:
        outputs = np.concatenate([np.array(reserved), ret], axis=1)
      else:
        outputs = ret
      indices = range(outputs.shape[1])
      t2 = time.time()
      writer.write(outputs.T, indices, allow_type_cast=True)
      t3 = time.time()
      sum_t0 += (t1 - t0)
      sum_t1 += (t2 - t1)
      sum_t2 += (t3 - t2)
      if i % 2 == 0:
        logging.info('progress: batch_num=%d sample_num=%d' %
                     (i + 1, (i + 1) * batch_size))
        logging.info('time_stats: read: %.2f predict: %.2f write: %.2f' %
                     (sum_t0, sum_t1, sum_t2))
      logging.info('Final_time_stats: read: %.2f predict: %.2f write: %.2f' %
                   (sum_t0, sum_t1, sum_t2))
    writer.close()
    reader.close()
    logging.info('Explain %s done.' % input_path)


def _get_input_fields_from_pipeline_config(model_path):
  pipeline_path = os.path.join(model_path, 'assets/pipeline.config')
  if not gfile.Exists(pipeline_path):
    logging.warning(
      '%s not exists, default values maybe inconsistent with the values used in training.'
      % pipeline_path)
    return {}, []
  pipeline_config = get_configs_from_pipeline_file(pipeline_path)
  data_config = pipeline_config.data_config
  label_fields = data_config.label_fields
  labels = {x for x in label_fields}
  if data_config.HasField('sample_weight'):
    labels.add(data_config.sample_weight)

  input_fields = data_config.input_fields
  input_fields_info = {
    input_field.input_name:
      (input_field.input_type, input_field.default_val)
    for input_field in input_fields if input_field.input_name not in labels
  }
  input_fields_list = [input_field.input_name for input_field in input_fields if input_field.input_name not in labels]
  return input_fields_info, input_fields_list


def search_pb(directory, use_latest=False):
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
    if use_latest:
      logging.info('find %d models: %s' % (len(dir_list), ','.join(dir_list)))
      dir_list = sorted(
        dir_list,
        key=lambda x: int(x.split('/')[(-2 if (x[-1] == '/') else -1)]))
      return dir_list[-1]
    else:
      raise ValueError('multiple saved model found in directory %s' %
                       directory)

  return dir_list[0]


# def create_explainer(model_path, use_latest=False):
#   if gfile.IsDirectory(model_path):
#     model_path = search_pb(model_path, use_latest)
#   else:
#     raise ValueError('model_path should be a directory, path:' + model_path)
#   pipeline_path = os.path.join(model_path, 'assets/pipeline.config')
#   if not gfile.Exists(pipeline_path):
#     logging.warning('%s not exists' % pipeline_path)
#     raise ValueError('%s not exists' % pipeline_path)
#
#   pipeline_config = get_configs_from_pipeline_file(pipeline_path)
#   input_type = pipeline_config.data_config.input_type
#   if input_type in {DatasetConfig.OdpsInput, DatasetConfig.OdpsInputV2, DatasetConfig.OdpsInputV3}:
#     return OdpsExplainer(model_path)
#   if input_type in {DatasetConfig.OdpsRTPInput, DatasetConfig.OdpsRTPInputV2}:
#     return OdpsRtpExplainer(model_path)
#   raise ValueError("currently unsupported input type: " + input_type)


def run(FLAGS):
  model_path = FLAGS.saved_model_dir
  if gfile.IsDirectory(model_path):
    model_path = search_pb(model_path, False)
  else:
    raise ValueError('model_path should be a directory, path:' + model_path)
  pipeline_path = os.path.join(model_path, 'assets/pipeline.config')
  if not gfile.Exists(pipeline_path):
    logging.warning('%s not exists' % pipeline_path)
    raise ValueError('%s not exists' % pipeline_path)

  gpu_options = tf.GPUOptions(allow_growth=True)
  session_config = tf.ConfigProto(
    gpu_options=gpu_options,
    allow_soft_placement=True)
  session = tf.Session(config=session_config)

  worker_count = len(FLAGS.worker_hosts.split(','))
  with DeepExplain(session=session) as de:
    e = OdpsRtpExplainer(de, model_path, 'deeplift')
    e.feature_importance(FLAGS.explain_tables if FLAGS.explain_tables else FLAGS.tables,
                         FLAGS.outputs,
                         reserved_cols=FLAGS.reserved_cols,
                         output_cols=FLAGS.output_cols,
                         batch_size=FLAGS.batch_size,
                         slice_id=FLAGS.task_index,
                         slice_num=worker_count)
