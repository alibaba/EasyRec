# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf

import easy_rec
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils.load_class import get_register_class_meta

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

_INPUT_CLASS_MAP = {}
_meta_type = get_register_class_meta(_INPUT_CLASS_MAP, have_abstract_class=True)


class Input(six.with_metaclass(_meta_type, object)):

  def __init__(self,
               data_config,
               feature_configs,
               input_path,
               task_index=0,
               task_num=1):
    self._data_config = data_config

    if self._data_config.auto_expand_input_fields:
      input_fields = [x for x in self._data_config.input_fields]
      while len(self._data_config.input_fields) > 0:
        self._data_config.input_fields.pop()
      for field in input_fields:
        tmp_names = config_util.auto_expand_names(field.input_name)
        for tmp_name in tmp_names:
          one_field = DatasetConfig.Field()
          one_field.CopyFrom(field)
          one_field.input_name = tmp_name
          self._data_config.input_fields.append(one_field)

    self._input_fields = [x.input_name for x in data_config.input_fields]
    self._input_field_types = [x.input_type for x in data_config.input_fields]
    self._input_field_defaults = [
        x.default_val for x in data_config.input_fields
    ]
    self._label_fields = list(data_config.label_fields)

    self._batch_size = data_config.batch_size
    self._prefetch_size = data_config.prefetch_size
    self._feature_configs = list(feature_configs)
    self._task_index = task_index
    self._task_num = task_num

    self._input_path = input_path

    # findout effective fields
    self._effective_fields = []
    for fc in self._feature_configs:
      for input_name in fc.input_names:
        assert input_name in self._input_fields, 'invalid input_name in %s' % str(
            fc)
        self._effective_fields.append(input_name)
    self._effective_fids = [
        self._input_fields.index(x) for x in self._effective_fields
    ]

    self._label_fids = [self._input_fields.index(x) for x in self._label_fields]

    # appended fields
    self._appended_fields = []

  @property
  def num_epochs(self):
    if self._data_config.num_epochs > 0:
      return self._data_config.num_epochs
    else:
      return None

  def get_tf_type(self, field_type):
    type_map = {
        DatasetConfig.INT32: tf.int32,
        DatasetConfig.INT64: tf.int64,
        DatasetConfig.STRING: tf.string,
        DatasetConfig.BOOL: tf.bool,
        DatasetConfig.FLOAT: tf.float32,
        DatasetConfig.DOUBLE: tf.double
    }
    assert field_type in type_map, 'invalid type: %s' % field_type
    return type_map[field_type]

  def get_type_defaults(self, field_type, default_val=''):
    type_defaults = {
        DatasetConfig.INT32: 0,
        DatasetConfig.INT64: 0,
        DatasetConfig.STRING: '',
        DatasetConfig.BOOL: False,
        DatasetConfig.FLOAT: 0.0,
        DatasetConfig.DOUBLE: 0.0
    }
    assert field_type in type_defaults, 'invalid type: %s' % field_type
    if default_val == '':
      default_val = type_defaults[field_type]
    if field_type in [DatasetConfig.INT32, DatasetConfig.INT64]:
      return int(default_val)
    elif field_type == DatasetConfig.STRING:
      return default_val
    elif field_type == DatasetConfig.BOOL:
      return default_val.lower() == 'true'
    elif field_type in [DatasetConfig.FLOAT]:
      return float(default_val)
    elif field_type in [DatasetConfig.DOUBLE]:
      return np.float64(default_val)

    return type_defaults[field_type]

  def create_multi_placeholders(self):
    inputs = {}
    for fid in self._effective_fids:
      ftype = self._input_field_types[fid]
      tf_type = self.get_tf_type(ftype)
      input_name = self._input_fields[fid]
      finput = tf.placeholder(tf_type, [None], name='input_%d' % fid)
      inputs[input_name] = finput
    features = self._preprocess(inputs)
    return inputs, features

  def create_placeholders(self):
    inputs_placeholder = tf.placeholder(tf.string, [None], name='features')
    input_vals = tf.string_split(
        inputs_placeholder, self._data_config.separator,
        skip_empty=False).values
    input_vals = tf.reshape(
        input_vals, [-1, len(self._input_fields) - 1], name='input_reshape')
    features = {}
    for fid in self._effective_fids:
      ftype = self._input_field_types[fid]
      tf_type = self.get_tf_type(ftype)
      input_name = self._input_fields[fid]
      if tf_type in [tf.float32, tf.double, tf.int32, tf.int64]:
        features[input_name] = tf.string_to_number(
            input_vals[:, fid - 1],
            tf_type,
            name='input_str_to_%s' % tf_type.name)
      else:
        features[input_name] = input_vals[:, fid - 1]
    features = self._preprocess(features)
    return {'features': inputs_placeholder}, features

  def create_embedding_placeholders(self):
    """Create serving placeholders with embedding lookup result as input.

    For large embedding serving on eas, the embedding lookup are done on eas distributedly,
    so it is not included in the graph.

    Return:
      dict of embedding_name to placeholders
    """
    inputs = {}
    dump_input_dir = easy_rec._global_config['dump_embedding_shape_dir']
    embed_input_desc_files = tf.gfile.Glob(
        os.path.join(dump_input_dir, 'input_layer_*.txt'))
    for one_file in embed_input_desc_files:
      with tf.gfile.GFile(one_file, 'r') as fin:
        for line_str in fin:
          shape_config = json.loads(line_str)
          input_name = shape_config['name']
          if input_name.endswith('_shared_embedding'):
            input_name = input_name[:-len('_shared_embedding')]
          elif input_name.endswith('_embedding'):
            input_name = input_name[:-len('_embedding')]
          if '_weighted_by' in input_name:
            input_name = input_name[:input_name.find('_weighted_by')]
          inputs[input_name] = tf.placeholder(
              tf.float32, shape_config['shape'], name=input_name)
    return inputs

  def _get_features(self, fields):
    field_dict = {x: fields[x] for x in self._effective_fields if x in fields}
    for k in self._appended_fields:
      field_dict[k] = fields[k]
    return field_dict

  def _get_labels(self, fields):
    return OrderedDict([(x, fields[x]) for x in self._label_fields])

  def _preprocess(self, field_dict):
    """Preprocess the feature columns.

    preprocess some feature columns, such as TagFeature or LookupFeature,
    it is expected to handle batch inputs and single input,
    it could be customized in subclasses

    Args:
      field_dict: string to tensor, tensors are dense,
          could be of shape [batch_size], or of shape []

    Returns:
      output_dict: some of the tensors are transformed into sparse tensors,
          such as input tensors of tag features and lookup features
    """
    parsed_dict = {}
    for fc in self._feature_configs:
      feature_name = fc.feature_name
      feature_type = fc.feature_type
      input_0 = fc.input_names[0]
      if feature_type == fc.TagFeature:
        input_0 = fc.input_names[0]
        field = field_dict[input_0]
        if len(field.get_shape()) == 0:
          field = tf.expand_dims(field, axis=0)
        parsed_dict[input_0] = tf.string_split(field, fc.separator)
        if not fc.HasField('hash_bucket_size'):
          vals = tf.string_to_number(
              parsed_dict[input_0].values,
              tf.int32,
              name='tag_fea_%s' % input_0)
          parsed_dict[input_0] = tf.sparse.SparseTensor(
              parsed_dict[input_0].indices, vals,
              parsed_dict[input_0].dense_shape)
        if len(fc.input_names) > 1:
          input_1 = fc.input_names[1]
          field = field_dict[input_1]
          if len(field.get_shape()) == 0:
            field = tf.expand_dims(field, axis=0)
          field = tf.string_split(field, fc.separator)
          field_vals = tf.string_to_number(
              field.values, tf.float32, name='tag_wgt_%s' % input_1)
          assert_op = tf.assert_equal(
              tf.shape(field_vals)[0],
              tf.shape(parsed_dict[input_0].values)[0],
              message='tag_feature_kv_size_not_eq')
          with tf.control_dependencies([assert_op]):
            field = tf.sparse.SparseTensor(field.indices, field_vals,
                                           field.dense_shape)
          parsed_dict[input_1] = field
      elif feature_type == fc.LookupFeature:
        assert feature_name is not None and feature_name != ''
        assert len(fc.input_names) == 2
        parsed_dict[feature_name] = self._lookup_preprocess(fc, field_dict)
      elif feature_type == fc.SequenceFeature:
        input_0 = fc.input_names[0]
        field = field_dict[input_0]
        parsed_dict[input_0] = tf.string_split(field, fc.separator)
      elif feature_type == fc.RawFeature:
        input_0 = fc.input_names[0]
        if field_dict[input_0].dtype == tf.string:
          parsed_dict[input_0] = tf.string_to_number(field_dict[input_0],
                                                     tf.float32)
        elif field_dict[input_0].dtype in [
            tf.int32, tf.int64, tf.double, tf.float32
        ]:
          parsed_dict[input_0] = tf.to_float(field_dict[input_0])
        else:
          assert False, 'invalid dtype[%s] for raw feature' % str(
              field_dict[input_0].dtype)
        if fc.max_val > fc.min_val:
          parsed_dict[input_0] = (parsed_dict[input_0] - fc.min_val) /\
                                 (fc.max_val - fc.min_val)
        if not fc.boundaries and fc.num_buckets <= 1:
          # may need by wide model and deep model to project
          # raw values to a vector, it maybe better implemented
          # by a ProjectionColumn later
          sample_num = tf.to_int64(tf.shape(parsed_dict[input_0])[0])
          indices_0 = tf.range(sample_num, dtype=tf.int64)
          indices_1 = tf.zeros([sample_num], dtype=tf.int64)
          indices_0 = tf.expand_dims(indices_0, axis=1)
          indices_1 = tf.expand_dims(indices_1, axis=1)
          indices = tf.concat([indices_0, indices_1], axis=1)

          parsed_dict[input_0 + '_raw_proj_id'] = tf.SparseTensor(
              indices=indices,
              values=tf.zeros_like(parsed_dict[input_0], dtype=tf.int32),
              dense_shape=[sample_num, 1])
          parsed_dict[input_0 + '_raw_proj_val'] = tf.SparseTensor(
              indices=indices,
              values=parsed_dict[input_0],
              dense_shape=[sample_num, 1])
          self._appended_fields.append(input_0 + '_raw_proj_id')
          self._appended_fields.append(input_0 + '_raw_proj_val')
      elif feature_type == fc.IdFeature:
        input_0 = fc.input_names[0]
        parsed_dict[input_0] = field_dict[input_0]
        if fc.HasField('hash_bucket_size'):
          if field_dict[input_0].dtype != tf.string:
            if field_dict[input_0].dtype in [tf.float32, tf.double]:
              assert fc.precision > 0, 'it is dangerous to convert float or double to string due to ' \
                                       'precision problem, it is suggested to convert them into string ' \
                                       'format during feature generalization before using EasyRec; ' \
                                       'if you really need to do so, please set precision (the number of ' \
                                       'decimal digits) carefully.'
            precision = None
            if field_dict[input_0].dtype in [tf.float32, tf.double]:
              if fc.precision > 0:
                precision = fc.precision
            # convert to string
            if 'as_string' in dir(tf.strings):
              parsed_dict[input_0] = tf.strings.as_string(
                  field_dict[input_0], precision=precision)
            else:
              parsed_dict[input_0] = tf.as_string(
                  field_dict[input_0], precision=precision)
      else:
        for input_name in fc.input_names:
          parsed_dict[input_name] = field_dict[input_name]

    for input_name in self._label_fields:
      if input_name not in field_dict:
        continue
      if field_dict[input_name].dtype == tf.string:
        parsed_dict[input_name] = tf.string_to_number(
            field_dict[input_name], tf.float32, name=input_name)
      else:
        assert field_dict[input_name].dtype in [
            tf.float32, tf.double, tf.int32, tf.int64
        ], 'invalid label dtype: %s' % str(field_dict[input_name].dtype)
        parsed_dict[input_name] = field_dict[input_name]

    return parsed_dict

  def _lookup_preprocess(self, fc, field_dict):
    """Preprocess function for lookup features.

    Args:
      fc: FeatureConfig
      field_dict: input dict

    Returns:
      output_dict: add { feature_name:SparseTensor} with
          other items similar as field_dict
    """
    max_sel_num = fc.lookup_max_sel_elem_num

    def _lookup(args, pad=True):
      one_key, one_map = args[0], args[1]
      if len(one_map.get_shape()) == 0:
        one_map = tf.expand_dims(one_map, axis=0)
      kv_map = tf.string_split(one_map, fc.separator).values
      kvs = tf.string_split(kv_map, fc.kv_separator)
      kvs = tf.reshape(kvs.values, [-1, 2], name='kv_split_reshape')
      keys, vals = kvs[:, 0], kvs[:, 1]
      sel_ids = tf.where(tf.equal(keys, one_key))
      sel_ids = tf.squeeze(sel_ids, axis=1)
      sel_vals = tf.gather(vals, sel_ids)
      if not pad:
        return sel_vals
      n = tf.shape(sel_vals)[0]
      sel_vals = tf.pad(sel_vals, [[0, max_sel_num - n]])
      len_msk = tf.sequence_mask(n, max_sel_num)
      indices = tf.range(max_sel_num, dtype=tf.int64)
      indices = indices * tf.to_int64(indices < tf.to_int64(n))
      return sel_vals, len_msk, indices

    key_field, map_field = fc.input_names[0], fc.input_names[1]
    key_fields, map_fields = field_dict[key_field], field_dict[map_field]
    if len(key_fields.get_shape()) == 0:
      vals = _lookup((key_fields, map_fields), False)
      n = tf.shape(vals)[0]
      n = tf.to_int64(n)
      indices_0 = tf.zeros([n], dtype=tf.int64)
      indices_1 = tf.range(0, n, dtype=tf.int64)
      indices = [
          tf.expand_dims(indices_0, axis=1),
          tf.expand_dims(indices_1, axis=1)
      ]
      indices = tf.concat(indices, axis=1)
      return tf.sparse.SparseTensor(indices, vals, [1, n])

    vals, masks, indices = tf.map_fn(
        _lookup, [key_fields, map_fields], dtype=(tf.string, tf.bool, tf.int64))
    batch_size = tf.to_int64(tf.shape(vals)[0])
    vals = tf.boolean_mask(vals, masks)
    indices_1 = tf.boolean_mask(indices, masks)
    indices_0 = tf.range(0, batch_size, dtype=tf.int64)
    indices_0 = tf.expand_dims(indices_0, axis=1)
    indices_0 = indices_0 + tf.zeros([1, max_sel_num], dtype=tf.int64)
    indices_0 = tf.boolean_mask(indices_0, masks)
    indices = tf.concat(
        [tf.expand_dims(indices_0, axis=1),
         tf.expand_dims(indices_1, axis=1)],
        axis=1)
    shapes = tf.stack([batch_size, tf.reduce_max(indices_1) + 1])
    return tf.sparse.SparseTensor(indices, vals, shapes)

  @abstractmethod
  def _build(self, mode, params):
    raise NotImplementedError

  def create_input(self, export_config=None):

    def _input_fn(mode=None, params=None, config=None):
      """Build input_fn for estimator.

      Args:
        mode: tf.estimator.ModeKeys.(TRAIN, EVAL, PREDICT)
            params: `dict` of hyper parameters, from Estimator
            config: tf.estimator.RunConfig instance

      Return:
        if mode is not None, return:
            features: inputs to the model.
            labels: groundtruth
        else, return:
            tf.estimator.export.ServingInputReceiver instance
      """
      if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
                  tf.estimator.ModeKeys.PREDICT):
        # build dataset from self._config.input_path
        dataset = self._build(mode, params)
        return dataset
      elif mode is None:  # serving_input_receiver_fn for export SavedModel
        if export_config.dump_embedding_shape:
          embed_inputs = self.create_embedding_placeholders()
          return tf.estimator.export.ServingInputReceiver(
              embed_inputs, embed_inputs)
        elif export_config.multi_placeholder:
          inputs, features = self.create_multi_placeholders()
          return tf.estimator.export.ServingInputReceiver(features, inputs)
        else:
          inputs, features = self.create_placeholders()
          return tf.estimator.export.ServingInputReceiver(features, inputs)

    return _input_fn
