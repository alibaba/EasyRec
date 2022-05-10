# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from itertools import accumulate
import logging
from collections import OrderedDict
from this import d

import tensorflow as tf
from tensorflow.python.client import device_lib

from easy_rec.python.compat import regularizers
from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.compat.feature_column import feature_column_v2
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.feature_column.feature_group import FeatureGroup
from easy_rec.python.layers import dnn
from easy_rec.python.layers import seq_input_layer
from easy_rec.python.layers import variational_dropout_layer
from easy_rec.python.layers.common_layers import text_cnn
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep
from easy_rec.python.utils import estimator_utils
from easy_rec.python.sok_adapter import sparse_operation_kit as sok

from easy_rec.python.compat.feature_column.feature_column import _SharedEmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import EmbeddingColumn  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class InputLayer(object):
  """Input Layer for generate input features.

  This class apply feature_columns to input tensors to generate wide features and deep features.
  """

  def __init__(self,
               feature_configs,
               feature_groups_config,
               variational_dropout_config=None,
               wide_output_dim=-1,
               use_embedding_variable=False,
               use_sok=False,
               sok_max_nnz=None,
               embedding_regularizer=None,
               kernel_regularizer=None,
               is_training=False):
    self._feature_groups = {
        x.group_name: FeatureGroup(x) for x in feature_groups_config
    }
    self._seq_feature_groups_config = []
    for x in feature_groups_config:
      for y in x.sequence_features:
        self._seq_feature_groups_config.append(y)
    self._group_name_to_seq_features = {
        x.group_name: x.sequence_features
        for x in feature_groups_config
        if len(x.sequence_features) > 0
    }
    self._seq_input_layer = None
    if len(self._seq_feature_groups_config) > 0:
      self._seq_input_layer = seq_input_layer.SeqInputLayer(
          feature_configs,
          self._seq_feature_groups_config,
          use_embedding_variable=use_embedding_variable)
    wide_and_deep_dict = self.get_wide_deep_dict()
    self._fc_parser = FeatureColumnParser(
        feature_configs,
        wide_and_deep_dict,
        wide_output_dim,
        use_embedding_variable=use_embedding_variable)

    self.use_sok = use_sok
    self.sok_max_nnz = sok_max_nnz
    self._embedding_regularizer = embedding_regularizer
    self._kernel_regularizer = kernel_regularizer
    self._is_training = is_training
    self._variational_dropout_config = variational_dropout_config

  def has_group(self, group_name):
    return group_name in self._feature_groups

  def target_attention(self, dnn_config, deep_fea, name):
    cur_id, hist_id_col, seq_len = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len']

    seq_max_len = tf.shape(hist_id_col)[1]
    emb_dim = hist_id_col.shape[2]

    cur_ids = tf.tile(cur_id, [1, seq_max_len])
    cur_ids = tf.reshape(cur_ids,
                         tf.shape(hist_id_col))  # (B, seq_max_len, emb_dim)

    din_net = tf.concat(
        [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
        axis=-1)  # (B, seq_max_len, emb_dim*4)

    din_layer = dnn.DNN(dnn_config, None, name, self._is_training)
    din_net = din_layer(din_net)
    scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

    seq_len = tf.expand_dims(seq_len, 1)
    mask = tf.sequence_mask(seq_len)
    padding = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(mask, scores, padding)  # [B, 1, seq_max_len]

    # Scale
    scores = tf.nn.softmax(scores)  # (B, 1, seq_max_len)
    hist_din_emb = tf.matmul(scores, hist_id_col)  # [B, 1, emb_dim]
    hist_din_emb = tf.reshape(hist_din_emb, [-1, emb_dim])  # [B, emb_dim]
    din_output = tf.concat([hist_din_emb, cur_id], axis=1)
    return din_output

  def call_seq_input_layer(self,
                           features,
                           all_seq_att_map_config,
                           feature_name_to_output_tensors=None):
    all_seq_fea = []
    # process all sequence features
    for seq_att_map_config in all_seq_att_map_config:
      group_name = seq_att_map_config.group_name
      allow_key_search = seq_att_map_config.allow_key_search
      seq_features = self._seq_input_layer(features, group_name,
                                           feature_name_to_output_tensors,
                                           allow_key_search)
      regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=[seq_features['key']])
      regularizers.apply_regularization(
          self._embedding_regularizer,
          weights_list=[seq_features['hist_seq_emb']])
      seq_dnn_config = None
      if seq_att_map_config.HasField('seq_dnn'):
        seq_dnn_config = seq_att_map_config.seq_dnn
      else:
        logging.info(
            'seq_dnn not set in seq_att_groups, will use default settings')
        # If not set seq_dnn, will use default settings
        from easy_rec.python.protos.dnn_pb2 import DNN
        seq_dnn_config = DNN()
        seq_dnn_config.hidden_units.extend([128, 64, 32, 1])
      cur_target_attention_name = 'seq_dnn' + group_name
      seq_fea = self.target_attention(
          seq_dnn_config, seq_features, name=cur_target_attention_name)
      all_seq_fea.append(seq_fea)
    # concat all seq_fea
    all_seq_fea = tf.concat(all_seq_fea, axis=1)
    return all_seq_fea

  def __call__(self, features, group_name, is_combine=True):
    """Get features by group_name.

    Args:
      features: input tensor dict
      group_name: feature_group name
      is_combine: whether to combine sequence features over the
          time dimension.

    Return:
      features: all features concatenate together
      group_features: list of features
      seq_features: list of sequence features, each element is a tuple:
          3 dimension embedding tensor (batch_size, max_seq_len, embedding_dimension),
          1 dimension sequence length tensor.
    """
    assert group_name in self._feature_groups, 'invalid group_name[%s], list: %s' % (
        group_name, ','.join([x for x in self._feature_groups]))
    feature_name_to_output_tensors = {}
    if group_name in self._group_name_to_seq_features:
      for seq_feature in self._group_name_to_seq_features[group_name]:
        for seq_att in seq_feature.seq_att_map:
          for k in seq_att.key:
            feature_name_to_output_tensors[k] = None
    if is_combine:
      concat_features, group_features = self.single_call_input_layer(
          features, group_name, is_combine, feature_name_to_output_tensors)
      if group_name in self._group_name_to_seq_features:
        seq_fea = self.call_seq_input_layer(
            features, self._group_name_to_seq_features[group_name],
            feature_name_to_output_tensors)
        concat_features = tf.concat([concat_features, seq_fea], axis=1)
      return concat_features, group_features
    else:
      if self._variational_dropout_config is not None:
        raise ValueError(
            'variational dropout is not supported in not combined mode now.')
      return self.single_call_input_layer(features, group_name, is_combine)

  def single_call_input_layer(self,
                              features,
                              group_name,
                              is_combine=True,
                              feature_name_to_output_tensors=None):
    """Get features by group_name.

    Args:
      features: input tensor dict
      group_name: feature_group name
      is_combine: whether to combine sequence features over the
          time dimension.
      feature_name_to_output_tensors: if set sequence_features, feature_name_to_output_tensors will
      take key tensors to reuse.

    Return:
      features: all features concatenate together
      group_features: list of features
      seq_features: list of sequence features, each element is a tuple:
          3 dimension embedding tensor (batch_size, max_seq_len, embedding_dimension),
          1 dimension sequence length tensor.
    """

    assert group_name in self._feature_groups, 'invalid group_name[%s], list: %s' % (
        group_name, ','.join([x for x in self._feature_groups]))
    feature_group = self._feature_groups[group_name]
    group_columns, group_seq_columns = feature_group.select_columns(
        self._fc_parser)
    if is_combine:
      cols_to_output_tensors = OrderedDict()
      if self.use_sok and group_name == 'sparse':
        output_features= self.sok_input_layer(features, group_columns,
                                              cols_to_output_tensors=cols_to_output_tensors)
      else:
        output_features = feature_column.input_layer(
            features,
            group_columns,
            cols_to_output_tensors=cols_to_output_tensors,
            feature_name_to_output_tensors=feature_name_to_output_tensors)
      embedding_reg_lst = [output_features]
      builder = feature_column._LazyBuilder(features)
      seq_features = []
      for column in sorted(group_seq_columns, key=lambda x: x.name):
        with tf.variable_scope(None, default_name=column._var_scope_name):
          seq_feature, seq_len = column._get_sequence_dense_tensor(builder)
          embedding_reg_lst.append(seq_feature)

          sequence_combiner = column.sequence_combiner
          if sequence_combiner is None:
            raise ValueError(
                'sequence_combiner is none, please set sequence_combiner or use TagFeature'
            )
          if sequence_combiner.WhichOneof('combiner') == 'attention':
            attn_logits = tf.layers.dense(
                inputs=seq_feature,
                units=1,
                kernel_regularizer=self._kernel_regularizer,
                use_bias=False,
                activation=None,
                name='attention')
            attn_logits = tf.squeeze(attn_logits, axis=-1)
            attn_logits_padding = tf.ones_like(attn_logits) * (-2**32 + 1)
            seq_mask = tf.sequence_mask(seq_len)
            attn_score = tf.nn.softmax(
                tf.where(seq_mask, attn_logits, attn_logits_padding))
            seq_feature = tf.reduce_sum(
                attn_score[:, :, tf.newaxis] * seq_feature, axis=1)
            seq_features.append(seq_feature)
            cols_to_output_tensors[column] = seq_feature
          elif sequence_combiner.WhichOneof('combiner') == 'text_cnn':
            num_filters = sequence_combiner.text_cnn.num_filters
            filter_sizes = sequence_combiner.text_cnn.filter_sizes
            cnn_feature = text_cnn(seq_feature, filter_sizes, num_filters)
            seq_features.append(cnn_feature)
            cols_to_output_tensors[column] = cnn_feature
          else:
            raise NotImplementedError
      if self._variational_dropout_config is not None:
        features_dimension = OrderedDict([
            (k.raw_name, int(v.shape[-1]))
            for k, v in cols_to_output_tensors.items()
        ])
        concat_features = tf.concat([output_features] + seq_features, axis=-1)
        variational_dropout = variational_dropout_layer.VariationalDropoutLayer(
            self._variational_dropout_config,
            features_dimension,
            self._is_training,
            name=group_name)
        concat_features = variational_dropout(concat_features)
        group_features = tf.split(
            concat_features, features_dimension.values(), axis=-1)
      else:
        concat_features = tf.concat([output_features] + seq_features, axis=-1)
        group_features = [cols_to_output_tensors[x] for x in group_columns] + \
                         [cols_to_output_tensors[x] for x in group_seq_columns]

      regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=embedding_reg_lst)
      return concat_features, group_features

    else:  # return sequence feature in raw format instead of combine them
      assert len(group_columns) == 0, \
          'there are none sequence columns: %s' % str(group_columns)
      builder = feature_column._LazyBuilder(features)
      seq_features = []
      embedding_reg_lst = []
      for fc in group_seq_columns:
        with tf.variable_scope('input_layer/' + fc.categorical_column.name):
          tmp_embedding, tmp_seq_len = fc._get_sequence_dense_tensor(builder)
          seq_features.append((tmp_embedding, tmp_seq_len))
          embedding_reg_lst.append(tmp_embedding)
      regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=embedding_reg_lst)
      return seq_features

  def sok_input_layer(self, features, group_columns,
                      cols_to_output_tensors=None):
    assert self.sok_max_nnz is not None
    max_nnz = self.sok_max_nnz
    allowd_column_types = [feature_column._EmbeddingColumn, feature_column_v2.EmbeddingColumn]
    _, workers = estimator_utils.get_task_index_and_num()

    # get gpu nums
    local_device_protos = device_lib.list_local_devices()
    gpus_num = len([x for x in local_device_protos if x.device_type == 'GPU'])

    try:
      assert len(group_columns) > 0
      assert type(group_columns[0]) in allowd_column_types
      emb_vec_size = group_columns[0].dimension
      combiner = group_columns[0].combiner

      bucket_sizes = []
      for column in group_columns:
        assert type(column) in allowd_column_types
        assert column.dimension == emb_vec_size
        assert column.combiner == combiner
        assert column.max_norm is None
        bucket_sizes.append(column.categorical_column._num_buckets)
    except AssertionError as e:
      logging.error("SOK requires all EmbeddingColumn have the same emb_dim and combiner. max_norm is not supported")
      raise e

    builder = feature_column._LazyBuilder(features)
    inputs = []
    for column in group_columns:
      sparse_tensor, _ = column.categorical_column._get_sparse_tensors(  # pylint: disable=protected-access
          builder, trainable=True)
      inputs.append(sparse_tensor)

    batch_size = inputs[0].dense_shape[0]
    sok_instance = tf.get_collection("SOK")[1]

    # slot_num = len(inputs) = len(group_columns)
    # try to combine slot_num SparseTensors into one big SparseTensor along the batch_size dimension, thus
    # need to accmulate indices and offset them.
    def sparse_sok():
      offetsed_values = []
      offsetted_indices = []

      for i, input_sparse in enumerate(inputs):
        offsetted_value = input_sparse.values + group_columns[i].categorical_column.hash_bucket_size
        offetsed_values.append(offsetted_value)

        offsetted_indice = input_sparse.indices + tf.stack([[i * batch_size, 0]])
        offsetted_indices.append(offsetted_indice)

      offetsed_values = tf.concat(offetsed_values, axis=0)
      offsetted_indices = tf.concat(offsetted_indices, axis=0)
      dense_shape = tf.concat([[batch_size * len(inputs)],
                              tf.constant(max_nnz, shape=(1,), dtype=tf.int64)], axis=0)

      combined_input = tf.SparseTensor(
        values=offetsed_values,
        indices=offsetted_indices,
        dense_shape=dense_shape
      )

      # output should be [batch_size, slot_num, emb_vec_size]
      return sok_instance(combined_input, training=self._is_training)

    def dense_sok():
      offetsed_values = []

      for i, input_sparse in enumerate(inputs):
        offsetted_value = input_sparse.values + group_columns[i].categorical_column.hash_bucket_size
        offetsed_values.append(offsetted_value)

      offetsed_values = tf.concat(offetsed_values, axis=0)

      return sok_instance(offetsed_values, training=self._is_training)

    output = sparse_sok()
    #output = dense_sok()

    for i, _ in enumerate(inputs):
      output_piece = tf.squeeze(
        tf.slice(output, [0, i, 0], [batch_size, 1, emb_vec_size]),
        axis=1)
      cols_to_output_tensors[group_columns[i]] = output_piece
    return output

  def get_wide_deep_dict(self):
    """Get wide or deep indicator for feature columns.

    Returns:
      dict of { feature_name : WideOrDeep }
    """
    wide_and_deep_dict = {}
    for fg_name in self._feature_groups.keys():
      fg = self._feature_groups[fg_name]
      tmp_dict = fg.wide_and_deep_dict
      for k in tmp_dict:
        v = tmp_dict[k]
        if k not in wide_and_deep_dict:
          wide_and_deep_dict[k] = v
        elif wide_and_deep_dict[k] != v:
          wide_and_deep_dict[k] = WideOrDeep.WIDE_AND_DEEP
        else:
          pass
    return wide_and_deep_dict
