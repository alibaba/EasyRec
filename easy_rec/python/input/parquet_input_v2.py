# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# import logging
import os

# import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops

from easy_rec.python.input.parquet_input import ParquetInput
from easy_rec.python.utils import conditional

# from easy_rec.python.utils.tf_utils import get_tf_type


class ParquetInputV2(ParquetInput):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None,
               **kwargs):
    super(ParquetInputV2,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config, **kwargs)
    self._need_pack = False

  def _predictor_preprocess(self, input_dict):
    # when the ParquetInputV2 is build from ParquetPredictorV2
    # the feature preprocess stage will be skipped.
    fea_dict = {}
    for k in input_dict:
      vals = input_dict[k]
      if isinstance(vals, tuple) and len(vals) == 2 and k != 'reserve':
        fea_dict[k + '/lens'] = vals[0]
        fea_dict[k + '/ids'] = vals[1]
      else:
        fea_dict[k] = vals
    return fea_dict

  def _to_fea_dict(self, input_dict):
    if self._is_predictor:
      fea_dict = self._predictor_preprocess(input_dict)
    else:
      fea_dict = self._preprocess(input_dict)

    output_dict = {'feature': fea_dict}

    lbl_dict = {}
    for lbl_name in self._label_fields:
      if lbl_name in input_dict:
        lbl_dict[lbl_name] = input_dict[lbl_name]

    if len(lbl_dict) > 0:
      output_dict['label'] = lbl_dict

    if self._reserve_fields is not None:
      output_dict['reserve'] = input_dict['reserve']

    return output_dict

  def add_fea_type_and_shape(self, out_types, out_shapes):
    # overload ParquetInput.build_type_and_shape
    for k in self._sparse_fea_names:
      out_types[k] = (tf.int32, tf.int64)
      out_shapes[k] = (tf.TensorShape([None]), tf.TensorShape([None]))
    for fc in self._dense_fea_cfgs:
      k = fc.input_names[0]
      out_types[k] = tf.float32
      out_shapes[k] = tf.TensorShape([None, fc.raw_input_dim])

  def _preprocess(self, inputs=None):
    features = {}
    placeholders = {}
    for fc in self._feature_configs:
      feature_name = fc.feature_name if fc.feature_name != '' else fc.input_names[
          0]
      feature_type = fc.feature_type
      if feature_type in [fc.IdFeature, fc.TagFeature]:
        input_name0 = fc.input_names[0]
        if inputs is not None:
          input_lens, input_vals = inputs[input_name0]
        else:
          if input_name0 in placeholders:
            input_lens, input_vals = placeholders[input_name0]
          else:
            input_vals = array_ops.placeholder(
                dtypes.int64, [None], name=input_name0 + '/ids')
            input_lens = array_ops.placeholder(
                dtypes.int64, [None], name=input_name0 + '/lens')
            placeholders[input_name0] = (input_lens, input_vals)
        if not self._has_ev:
          if fc.num_buckets > 0:
            input_vals = input_vals % fc.num_buckets
          else:
            input_vals = string_ops.as_string(input_vals)
        features[feature_name] = tf.RaggedTensor.from_row_lengths(
            values=input_vals, row_lengths=input_lens)
      elif feature_type in [fc.RawFeature]:
        input_name0 = fc.input_names[0]
        if inputs is not None:
          input_vals = inputs[input_name0]
        else:
          if input_name0 in placeholders:
            input_vals = placeholders[input_name0]
          else:
            if fc.raw_input_dim > 1:
              input_vals = array_ops.placeholder(
                  dtypes.float32, [None, fc.raw_input_dim], name=input_name0)
            else:
              input_vals = array_ops.placeholder(
                  dtypes.float32, [None], name=input_name0)
            placeholders[input_name0] = input_vals
        features[feature_name] = input_vals
      else:
        assert False, 'feature_type[%s] not supported' % str(feature_type)

    if inputs is not None:
      return features
    else:
      inputs = {}
      for key in placeholders:
        vals = placeholders[key]
        if isinstance(vals, tuple):
          inputs[key + '/lens'] = vals[0]
          inputs[key + '/ids'] = vals[1]
        else:
          inputs[key] = vals
      return features, inputs

  def _get_for_predictor(self, fea_dict):
    # called by ParquetInputV2._build, format:
    # {
    #   "feature": {"user_id/ids":..., "user_id/lens":..., ... },
    #   "reserve": {"sample_id":..., ...}
    # }
    return fea_dict

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
        self._mode = mode
        dataset = self._build(mode, params)
        return dataset
      elif mode is None:  # serving_input_receiver_fn for export SavedModel
        place_on_cpu = os.getenv('place_embedding_on_cpu')
        place_on_cpu = bool(place_on_cpu) if place_on_cpu else False
        with conditional(place_on_cpu, ops.device('/CPU:0')):
          features, inputs = self._preprocess()
        return tf.estimator.export.ServingInputReceiver(features, inputs)

    _input_fn.input_creator = self
    return _input_fn
