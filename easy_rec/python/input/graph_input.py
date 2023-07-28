# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from easy_rec.python.core import sampler
from easy_rec.python.input.input import Input
from easy_rec.python.utils import graph_utils

try:
  import graphlearn as gl
except ImportError:
  logging.error(
      'GraphLearn is not installed. You can install it by "pip install https://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/graphlearn-0.7-cp27-cp27mu-linux_x86_64.whl"'  # noqa: E501
  )
  sys.exit(1)

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

class GraphInput(Input):
  _g = None
  def __init__(self,
               data_config,
               feature_configs,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False):
    super(GraphInput, self).__init__(data_config, feature_configs, input_path,
                                     task_index, task_num, check_mode)
    
    gl.set_shuffle_buffer_size(102400000)
    if input_path:
      if GraphInput._g is None:
        if self._data_config.HasField('ultra_gcn_sampler'):
          GraphInput._g = gl.Graph()\
            .node(tf.compat.as_str(input_path.user_node_input), node_type="u",
                decoder=gl.Decoder(attr_types=['int']))\
            .node(tf.compat.as_str(input_path.item_node_input), node_type="i",
                decoder=gl.Decoder(attr_types=['int']))\
            .edge(tf.compat.as_str(input_path.u2i_edge_input), edge_type=("u", "i", "u-i"),
                decoder=gl.Decoder(weighted=False), directed=False)\
            .edge(tf.compat.as_str(input_path.i2i_edge_input), edge_type=("i", "i", "i-i"),
                decoder=gl.Decoder(weighted=True), directed=True)
        graph_utils.graph_init(GraphInput._g,os.environ.get('TF_CONFIG', None))
  
  def _sample_generator_ultragcn(self):
    
    def ultragcn_sampler():
      epoch_id = 0
      while self.num_epochs is None or epoch_id < self.num_epochs:
        if self._mode == tf.estimator.ModeKeys.TRAIN:
          self.edge_sampler = GraphInput._g.edge_sampler("u-i", self._batch_size, strategy="shuffle")
          self.u_sampler = GraphInput._g.node_sampler("u", self._batch_size, strategy="by_order")
          self.i_sampler = GraphInput._g.node_sampler("i", self._batch_size, strategy="by_order")
          self.i2i_nbr_sampler = GraphInput._g.neighbor_sampler("i-i", self._nbr_num, strategy="topk")
          self.neg_sampler = GraphInput._g.negative_sampler("u-i", self._neg_num, "random")
        else:
          self.edge_sampler = GraphInput._g.edge_sampler("u-i", self._batch_size, strategy="shuffle")
          self.neg_sampler = GraphInput._g.negative_sampler("u-i", self._neg_num, "random")
          self.u_sampler = GraphInput._g.node_sampler("u", self._batch_size, strategy="by_order")
          self.i_sampler = GraphInput._g.node_sampler("i", self._batch_size, strategy="by_order")
          self.i2i_nbr_sampler = GraphInput._g.neighbor_sampler("i-i", self._nbr_num, strategy="topk")

        while True:
          try:
            samples=[]
            edges = self.edge_sampler.get()
            neg_items = self.neg_sampler.get(edges.src_ids)
            nbr_items = self.i2i_nbr_sampler.get(edges.dst_ids)
            samples.append(edges.src_ids) # user ids
            samples.append(self._g.out_degrees(edges.src_ids, 'u-i')) # user degrees
            samples.append(edges.dst_ids) # item ids
            samples.append(self._g.out_degrees(edges.dst_ids, 'u-i_reverse')) # item degrees
            samples.append(nbr_items.layer_nodes(1).ids) # nbr item ids
            samples.append(nbr_items.layer_edges(1).weights) # nbr item weight.
            samples.append(neg_items.ids) # neg item ids
            yield(tuple(samples))
          except gl.OutOfRangeError:
            break
        if self._mode != tf.estimator.ModeKeys.TRAIN:
          break
        epoch_id += 1

    self._nbr_num = self._data_config.ultra_gcn_sampler.nbr_num
    self._neg_num = self._data_config.ultra_gcn_sampler.neg_num
    output_types = [tf.int64, tf.float32, tf.int64, tf.float32,
      tf.int64, tf.float32, tf.int64]
    # user ids, user degrees, item ids, item degrees, nbr item ids, nbr item weight, neg item ids
    output_shapes = [tf.TensorShape([None]),
                      tf.TensorShape([None]),
                      tf.TensorShape([None]),
                      tf.TensorShape([None]),
                      tf.TensorShape([None, self._nbr_num]),
                      tf.TensorShape([None, self._nbr_num]),
                      tf.TensorShape([None, self._neg_num])]
    
    dataset = tf.data.Dataset.from_generator(
        ultragcn_sampler,
        output_types=tuple(output_types),
        output_shapes=tuple(output_shapes))
    return dataset

  def _to_fea_dict(self, *features):
    fea_dict_= {'features': []}
    fea_dict_['features'] = features
    return fea_dict_

  def _get_features(self, field_dict_groups):
    return {
      'features': field_dict_groups['features']
    }

  def _get_labels(self, field_dict):
    return {
    }

  def _preprocess(self, field_dict):
    """Preprocess the feature columns.

    preprocess some feature columns, such as TagFeature or LookupFeature,
    it is expected to handle batch inputs and single input,
    it could be customized in subclasses

    Args:
      field_dict: string to tensor, tensors are dense,
          could be of shape [batch_size], [batch_size, None], or of shape []

    Returns:
      output_dict: some of the tensors are transformed into sparse tensors,
          such as input tensors of tag features and lookup features
    """
    parsed_dict = {}

    if self._sampler is not None and self._mode != tf.estimator.ModeKeys.PREDICT:
      if self._mode != tf.estimator.ModeKeys.TRAIN:
        self._sampler.set_eval_num_sample()
      sampler_type = self._data_config.WhichOneof('sampler')
      sampler_config = getattr(self._data_config, sampler_type)
      item_ids = field_dict[sampler_config.item_id_field]
      if sampler_type in ['negative_sampler', 'negative_sampler_in_memory']:
        sampled = self._sampler.get(item_ids)
      elif sampler_type == 'negative_sampler_v2':
        user_ids = field_dict[sampler_config.user_id_field]
        sampled = self._sampler.get(user_ids, item_ids)
      elif sampler_type.startswith('hard_negative_sampler'):
        user_ids = field_dict[sampler_config.user_id_field]
        sampled = self._sampler.get(user_ids, item_ids)
      else:
        raise ValueError('Unknown sampler %s' % sampler_type)
      for k, v in sampled.items():
        if k in field_dict:
          field_dict[k] = tf.concat([field_dict[k], v], axis=0)
        else:
          print('appended fields: %s' % k)
          parsed_dict[k] = v
          self._appended_fields.append(k)

    for fc in self._feature_configs:
      feature_name = fc.feature_name
      feature_type = fc.feature_type
      input_0 = fc.input_names[0]
      if feature_type == fc.TagFeature:
        input_0 = fc.input_names[0]
        field = field_dict[input_0]
        # Construct the output of TagFeature according to the dimension of field_dict.
        # When the input field exceeds 2 dimensions, convert TagFeature to 2D output.
        if len(field.get_shape()) < 2 or field.get_shape()[-1] == 1:
          if len(field.get_shape()) == 0:
            field = tf.expand_dims(field, axis=0)
          elif len(field.get_shape()) == 2:
            field = tf.squeeze(field, axis=-1)
          if fc.HasField('kv_separator') and len(fc.input_names) > 1:
            assert False, 'Tag Feature Error, ' \
                          'Cannot set kv_separator and multi input_names in one feature config. Feature: %s.' % input_0
          parsed_dict[input_0] = tf.string_split(field, fc.separator)
          if fc.HasField('kv_separator'):
            indices = parsed_dict[input_0].indices
            tmp_kvs = parsed_dict[input_0].values
            tmp_kvs = tf.string_split(
                tmp_kvs, fc.kv_separator, skip_empty=False)
            tmp_kvs = tf.reshape(tmp_kvs.values, [-1, 2])
            tmp_ks, tmp_vs = tmp_kvs[:, 0], tmp_kvs[:, 1]

            check_list = [
                tf.py_func(
                    check_string_to_number, [tmp_vs, input_0], Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              tmp_vs = tf.string_to_number(
                  tmp_vs, tf.float32, name='kv_tag_wgt_str_2_flt_%s' % input_0)
            parsed_dict[input_0] = tf.sparse.SparseTensor(
                indices, tmp_ks, parsed_dict[input_0].dense_shape)
            input_wgt = input_0 + '_WEIGHT'
            parsed_dict[input_wgt] = tf.sparse.SparseTensor(
                indices, tmp_vs, parsed_dict[input_0].dense_shape)
            self._appended_fields.append(input_wgt)
          if not fc.HasField('hash_bucket_size'):
            check_list = [
                tf.py_func(
                    check_string_to_number,
                    [parsed_dict[input_0].values, input_0],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
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
            check_list = [
                tf.py_func(
                    check_string_to_number, [field.values, input_1],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              field_vals = tf.string_to_number(
                  field.values,
                  tf.float32,
                  name='tag_wgt_str_2_flt_%s' % input_1)
            assert_op = tf.assert_equal(
                tf.shape(field_vals)[0],
                tf.shape(parsed_dict[input_0].values)[0],
                message='TagFeature Error: The size of %s not equal to the size of %s. Please check input: %s and %s.'
                % (input_0, input_1, input_0, input_1))
            with tf.control_dependencies([assert_op]):
              field = tf.sparse.SparseTensor(field.indices,
                                             tf.identity(field_vals),
                                             field.dense_shape)
            parsed_dict[input_1] = field
        else:
          parsed_dict[input_0] = field_dict[input_0]
          if len(fc.input_names) > 1:
            input_1 = fc.input_names[1]
            parsed_dict[input_1] = field_dict[input_1]
      elif feature_type == fc.LookupFeature:
        assert feature_name is not None and feature_name != ''
        assert len(fc.input_names) == 2
        parsed_dict[feature_name] = self._lookup_preprocess(fc, field_dict)
      elif feature_type == fc.SequenceFeature:
        input_0 = fc.input_names[0]
        field = field_dict[input_0]
        sub_feature_type = fc.sub_feature_type
        # Construct the output of SeqFeature according to the dimension of field_dict.
        # When the input field exceeds 2 dimensions, convert SeqFeature to 2D output.
        if len(field.get_shape()) < 2:
          parsed_dict[input_0] = tf.strings.split(field, fc.separator)
          if fc.HasField('seq_multi_sep'):
            indices = parsed_dict[input_0].indices
            values = parsed_dict[input_0].values
            multi_vals = tf.string_split(values, fc.seq_multi_sep)
            indices_1 = multi_vals.indices
            indices = tf.gather(indices, indices_1[:, 0])
            out_indices = tf.concat([indices, indices_1[:, 1:]], axis=1)
            # 3 dimensional sparse tensor
            out_shape = tf.concat(
                [parsed_dict[input_0].dense_shape, multi_vals.dense_shape[1:]],
                axis=0)
            parsed_dict[input_0] = tf.sparse.SparseTensor(
                out_indices, multi_vals.values, out_shape)
          if (fc.num_buckets > 1 and fc.max_val == fc.min_val):
            check_list = [
                tf.py_func(
                    check_string_to_number,
                    [parsed_dict[input_0].values, input_0],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              parsed_dict[input_0] = tf.sparse.SparseTensor(
                  parsed_dict[input_0].indices,
                  tf.string_to_number(
                      parsed_dict[input_0].values,
                      tf.int64,
                      name='sequence_str_2_int_%s' % input_0),
                  parsed_dict[input_0].dense_shape)
          elif sub_feature_type == fc.RawFeature:
            check_list = [
                tf.py_func(
                    check_string_to_number,
                    [parsed_dict[input_0].values, input_0],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              parsed_dict[input_0] = tf.sparse.SparseTensor(
                  parsed_dict[input_0].indices,
                  tf.string_to_number(
                      parsed_dict[input_0].values,
                      tf.float32,
                      name='sequence_str_2_float_%s' % input_0),
                  parsed_dict[input_0].dense_shape)
          if fc.num_buckets > 1 and fc.max_val > fc.min_val:
            normalized_values = (parsed_dict[input_0].values - fc.min_val) / (
                fc.max_val - fc.min_val)
            parsed_dict[input_0] = tf.sparse.SparseTensor(
                parsed_dict[input_0].indices, normalized_values,
                parsed_dict[input_0].dense_shape)
        else:
          parsed_dict[input_0] = field
        if not fc.boundaries and fc.num_buckets <= 1 and fc.hash_bucket_size <= 0 and \
            self._data_config.sample_weight != input_0 and sub_feature_type == fc.RawFeature and \
            fc.raw_input_dim == 1:
          # may need by wide model and deep model to project
          # raw values to a vector, it maybe better implemented
          # by a ProjectionColumn later
          logging.info(
              'Not set boundaries or num_buckets or hash_bucket_size, %s will process as two dimension raw feature'
              % input_0)
          parsed_dict[input_0] = tf.sparse_to_dense(
              parsed_dict[input_0].indices,
              [tf.shape(parsed_dict[input_0])[0], fc.sequence_length],
              parsed_dict[input_0].values)
          sample_num = tf.to_int64(tf.shape(parsed_dict[input_0])[0])
          indices_0 = tf.range(sample_num, dtype=tf.int64)
          indices_1 = tf.range(fc.sequence_length, dtype=tf.int64)
          indices_0 = indices_0[:, None]
          indices_1 = indices_1[None, :]
          indices_0 = tf.tile(indices_0, [1, fc.sequence_length])
          indices_1 = tf.tile(indices_1, [sample_num, 1])
          indices_0 = tf.reshape(indices_0, [-1, 1])
          indices_1 = tf.reshape(indices_1, [-1, 1])
          indices = tf.concat([indices_0, indices_1], axis=1)
          parsed_dict[input_0 + '_raw_proj_id'] = tf.SparseTensor(
              indices=indices,
              values=indices_1[:, 0],
              dense_shape=[sample_num, fc.sequence_length])
          parsed_dict[input_0 + '_raw_proj_val'] = tf.SparseTensor(
              indices=indices,
              values=tf.reshape(parsed_dict[input_0], [-1]),
              dense_shape=[sample_num, fc.sequence_length])
          self._appended_fields.append(input_0 + '_raw_proj_id')
          self._appended_fields.append(input_0 + '_raw_proj_val')
        elif not fc.boundaries and fc.num_buckets <= 1 and fc.hash_bucket_size <= 0 and \
            self._data_config.sample_weight != input_0 and sub_feature_type == fc.RawFeature and \
            fc.raw_input_dim > 1:
          # for 3 dimension sequence feature input.
          # may need by wide model and deep model to project
          # raw values to a vector, it maybe better implemented
          # by a ProjectionColumn later
          logging.info(
              'Not set boundaries or num_buckets or hash_bucket_size, %s will process as three dimension raw feature'
              % input_0)
          parsed_dict[input_0] = tf.sparse_to_dense(
              parsed_dict[input_0].indices, [
                  tf.shape(parsed_dict[input_0])[0], fc.sequence_length,
                  fc.raw_input_dim
              ], parsed_dict[input_0].values)
          sample_num = tf.to_int64(tf.shape(parsed_dict[input_0])[0])
          indices_0 = tf.range(sample_num, dtype=tf.int64)
          indices_1 = tf.range(fc.sequence_length, dtype=tf.int64)
          indices_2 = tf.range(fc.raw_input_dim, dtype=tf.int64)
          indices_0 = indices_0[:, None, None]
          indices_1 = indices_1[None, :, None]
          indices_2 = indices_2[None, None, :]
          indices_0 = tf.tile(indices_0,
                              [1, fc.sequence_length, fc.raw_input_dim])
          indices_1 = tf.tile(indices_1, [sample_num, 1, fc.raw_input_dim])
          indices_2 = tf.tile(indices_2, [sample_num, fc.sequence_length, 1])
          indices_0 = tf.reshape(indices_0, [-1, 1])
          indices_1 = tf.reshape(indices_1, [-1, 1])
          indices_2 = tf.reshape(indices_2, [-1, 1])
          indices = tf.concat([indices_0, indices_1, indices_2], axis=1)

          parsed_dict[input_0 + '_raw_proj_id'] = tf.SparseTensor(
              indices=indices,
              values=indices_1[:, 0],
              dense_shape=[sample_num, fc.sequence_length, fc.raw_input_dim])
          parsed_dict[input_0 + '_raw_proj_val'] = tf.SparseTensor(
              indices=indices,
              values=tf.reshape(parsed_dict[input_0], [-1]),
              dense_shape=[sample_num, fc.sequence_length, fc.raw_input_dim])
          self._appended_fields.append(input_0 + '_raw_proj_id')
          self._appended_fields.append(input_0 + '_raw_proj_val')
      elif feature_type == fc.RawFeature:
        input_0 = fc.input_names[0]
        if field_dict[input_0].dtype == tf.string:
          if fc.raw_input_dim > 1:
            check_list = [
                tf.py_func(
                    check_split, [
                        field_dict[input_0], fc.separator, fc.raw_input_dim,
                        input_0
                    ],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              tmp_fea = tf.string_split(field_dict[input_0], fc.separator)
            check_list = [
                tf.py_func(
                    check_string_to_number, [tmp_fea.values, input_0],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              tmp_vals = tf.string_to_number(
                  tmp_fea.values,
                  tf.float32,
                  name='multi_raw_fea_to_flt_%s' % input_0)
            parsed_dict[input_0] = tf.sparse_to_dense(
                tmp_fea.indices,
                [tf.shape(field_dict[input_0])[0], fc.raw_input_dim],
                tmp_vals,
                default_value=0)
          else:
            check_list = [
                tf.py_func(
                    check_string_to_number, [field_dict[input_0], input_0],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              parsed_dict[input_0] = tf.string_to_number(
                  field_dict[input_0], tf.float32)
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
        if not fc.boundaries and fc.num_buckets <= 1 and \
            self._data_config.sample_weight != input_0:
          # may need by wide model and deep model to project
          # raw values to a vector, it maybe better implemented
          # by a ProjectionColumn later
          sample_num = tf.to_int64(tf.shape(parsed_dict[input_0])[0])
          indices_0 = tf.range(sample_num, dtype=tf.int64)
          indices_1 = tf.range(fc.raw_input_dim, dtype=tf.int64)
          indices_0 = indices_0[:, None]
          indices_1 = indices_1[None, :]
          indices_0 = tf.tile(indices_0, [1, fc.raw_input_dim])
          indices_1 = tf.tile(indices_1, [sample_num, 1])
          indices_0 = tf.reshape(indices_0, [-1, 1])
          indices_1 = tf.reshape(indices_1, [-1, 1])
          indices = tf.concat([indices_0, indices_1], axis=1)

          parsed_dict[input_0 + '_raw_proj_id'] = tf.SparseTensor(
              indices=indices,
              values=indices_1[:, 0],
              dense_shape=[sample_num, fc.raw_input_dim])
          parsed_dict[input_0 + '_raw_proj_val'] = tf.SparseTensor(
              indices=indices,
              values=tf.reshape(parsed_dict[input_0], [-1]),
              dense_shape=[sample_num, fc.raw_input_dim])
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
        elif fc.num_buckets > 0:
          if parsed_dict[input_0].dtype == tf.string:
            check_list = [
                tf.py_func(
                    check_string_to_number, [parsed_dict[input_0], input_0],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              parsed_dict[input_0] = tf.string_to_number(
                  parsed_dict[input_0], tf.int32, name='%s_str_2_int' % input_0)
      elif feature_type == fc.ExprFeature:
        fea_name = fc.feature_name
        prefix = 'expr_'
        for input_name in fc.input_names:
          new_input_name = prefix + input_name
          if field_dict[input_name].dtype == tf.string:
            check_list = [
                tf.py_func(
                    check_string_to_number,
                    [field_dict[input_name], input_name],
                    Tout=tf.bool)
            ] if self._check_mode else []
            with tf.control_dependencies(check_list):
              parsed_dict[new_input_name] = tf.string_to_number(
                  field_dict[input_name],
                  tf.float64,
                  name='%s_str_2_int_for_expr' % new_input_name)
          elif field_dict[input_name].dtype in [
              tf.int32, tf.int64, tf.double, tf.float32
          ]:
            parsed_dict[new_input_name] = tf.cast(field_dict[input_name],
                                                  tf.float64)
          else:
            assert False, 'invalid input dtype[%s] for expr feature' % str(
                field_dict[input_name].dtype)

        expression = get_expression(
            fc.expression, fc.input_names, prefix=prefix)
        logging.info('expression: %s' % expression)
        parsed_dict[fea_name] = eval(expression)
        self._appended_fields.append(fea_name)
      else:
        for input_name in fc.input_names:
          parsed_dict[input_name] = field_dict[input_name]

    for input_id, input_name in enumerate(self._label_fields):
      if input_name not in field_dict:
        continue
      if field_dict[input_name].dtype == tf.string:
        if self._label_dim[input_id] > 1:
          logging.info('will split labels[%d]=%s' % (input_id, input_name))
          check_list = [
              tf.py_func(
                  check_split, [
                      field_dict[input_name], self._label_sep[input_id],
                      self._label_dim[input_id], input_name
                  ],
                  Tout=tf.bool)
          ] if self._check_mode else []
          with tf.control_dependencies(check_list):
            parsed_dict[input_name] = tf.string_split(
                field_dict[input_name], self._label_sep[input_id]).values
            parsed_dict[input_name] = tf.reshape(
                parsed_dict[input_name], [-1, self._label_dim[input_id]])
        else:
          parsed_dict[input_name] = field_dict[input_name]
        check_list = [
            tf.py_func(
                check_string_to_number, [parsed_dict[input_name], input_name],
                Tout=tf.bool)
        ] if self._check_mode else []
        with tf.control_dependencies(check_list):
          parsed_dict[input_name] = tf.string_to_number(
              parsed_dict[input_name], tf.float32, name=input_name)
      else:
        assert field_dict[input_name].dtype in [
            tf.float32, tf.double, tf.int32, tf.int64
        ], 'invalid label dtype: %s' % str(field_dict[input_name].dtype)
        parsed_dict[input_name] = field_dict[input_name]

    if self._data_config.HasField('sample_weight'):
      if self._mode != tf.estimator.ModeKeys.PREDICT:
        parsed_dict[constant.SAMPLE_WEIGHT] = field_dict[
            self._data_config.sample_weight]
    
    if self._data_config.input_type == 19:
      parsed_dict = {}
      for fd in field_dict:
        parsed_dict[fd] = field_dict[fd]

    return parsed_dict

  def _build(self, mode, params):
    """Build graph dataset input for estimator.

    Args:
      mode: tf.estimator.ModeKeys.(TRAIN, EVAL, PREDICT)
      params: `dict` of hyper parameters, from Estimator

    Return:
      dataset: dataset for graph models.
    """
    if self._data_config.HasField('ultra_gcn_sampler'):
      dataset = self._sample_generator_ultragcn()

    # transform list to feature dict
    dataset = dataset.map(map_func=self._to_fea_dict)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    
    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
   
    return dataset
 
  def __del__(self):
    GraphInput._g.close()
