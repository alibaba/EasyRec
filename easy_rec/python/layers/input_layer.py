# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from google.protobuf import message
import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.feature_column.feature_group import FeatureGroup
from easy_rec.python.layers import variational_dropout_layer
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep

from easy_rec.python.compat.feature_column.feature_column import _SharedEmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import EmbeddingColumn  # NOQA
from easy_rec.python.layers import seq_input_layer
from easy_rec.python.layers import dnn
from easy_rec.python.compat import regularizers
if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class InputLayer(object):
  """Input Layer for generate input features.

  This class apply feature_columns to input tensors to generate wide features and deep features.
  """

  def __init__(self,
               feature_configs,
               feature_groups_config,
               seq_feature_groups_config=None,
               variational_dropout_config=None,
               wide_output_dim=-1,
               use_embedding_variable=False,
               embedding_regularizer=None,
               kernel_regularizer=None,
               is_training=False):
    self._feature_groups = {
        x.group_name: FeatureGroup(x) for x in feature_groups_config
    }
    self._seq_feature_groups = {}
    self._seq_input_layer = None
    if seq_feature_groups_config and len(seq_feature_groups_config) > 0:
      self._seq_feature_groups = {x.group_name:x for x in seq_feature_groups_config}
      self._seq_input_layer = seq_input_layer.SeqInputLayer(feature_configs, seq_feature_groups_config)
    wide_and_deep_dict = self.get_wide_deep_dict()
    self._fc_parser = FeatureColumnParser(
        feature_configs,
        wide_and_deep_dict,
        wide_output_dim,
        use_embedding_variable=use_embedding_variable)

    self._embedding_regularizer = embedding_regularizer
    self._kernel_regularizer = kernel_regularizer
    self._is_training = is_training
    self._variational_dropout_config = variational_dropout_config

  def has_group(self, group_name):
    return group_name in self._feature_groups

  def din(self, dnn_config, deep_fea, name):
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
  
  def __call_seq_input_(self, features, group_name, only_sequence_feature=False):
    seq_features = self._seq_input_layer(features, group_name)
    if only_sequence_feature:
      return seq_features
    regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=[seq_features['key']])
    regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=[seq_features['hist_seq_emb']])
    seq_att_map_config = self._seq_feature_groups[group_name]
    seq_dnn_config = None
    if seq_att_map_config.HasField("seq_dnn"):
      seq_dnn_config = seq_att_map_config.seq_dnn
    else:
      logging.info("seq_dnn not set in seq_att_groups, will use default settings")
      from easy_rec.python.protos.dnn_pb2 import DNN
      seq_dnn_config = DNN()
      seq_dnn_config.hidden_units.extend([128, 64, 32, 1])
    seq_fea = self.din(seq_dnn_config, seq_features, name='seq_dnn')
    return seq_fea

  
  def __call__(self, features, *group_names, **kwargs):
    is_combine = True
    exclude_group_names = []
    only_sequence_feature = False
    logging.info("input_group_names = {0}".format(group_names))
    if "is_combine" in kwargs:
      is_combine = kwargs["is_combine"]
    if "exclude_group_names" in kwargs:
      exclude_group_names = kwargs["exclude_group_names"]
    if "only_sequence_feature" in kwargs:
      only_sequence_feature = kwargs['only_sequence_feature']
    # if group_names is empty, we will use all feature groups include seq_att_map in config
    if len(group_names) == 0:
      group_names = list(self._feature_groups.keys()) + list(self._seq_feature_groups.keys())
      logging.info("use_group_names = {0}".format(group_names))
    if is_combine:
      all_concat_features, all_group_features = [], []
      for g in group_names:
        if g in exclude_group_names:
          continue
        if g in self._feature_groups:
          concat_features, group_features = self.__single_call__(features, g, is_combine)
          all_concat_features.append(concat_features)
          all_group_features.extend(group_features)
        else:
          logging.info("use sequence feature")
          seq_fea = self.__call_seq_input_(features, g, only_sequence_feature)
          all_concat_features.append(seq_fea)
      if len(all_concat_features) > 1:
        all_seq_concat_features = tf.concat(all_concat_features, axis=1)
      else:
        all_seq_concat_features = all_concat_features[0]
      return all_seq_concat_features, all_group_features
    else:
      all_seq_features = []
      for g in group_names:
        if g in exclude_group_names:
          continue
        if g in self._feature_groups:
          seq_features = self.__single_call__(features, g, is_combine)
          all_seq_features.extend(seq_features)
        else:
          seq_fea = self.__call_seq_input_(features, g, only_sequence_feature)
          all_seq_features.extend(seq_fea)  
      return all_seq_features    
      

  def __single_call__(self, features, group_name, is_combine=True):
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
    feature_group = self._feature_groups[group_name]
    group_columns, group_seq_columns = feature_group.select_columns(
        self._fc_parser)
    if is_combine:  # return sequence features in combined format
      cols_to_output_tensors = {}
      output_features = feature_column.input_layer(
          features,
          group_columns,
          cols_to_output_tensors=cols_to_output_tensors)
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
          else:
            raise NotImplementedError
      if self._variational_dropout_config is not None:
        variational_dropout = variational_dropout_layer.VariationalDropoutLayer(
            self._variational_dropout_config, group_columns, self._is_training)
        noisy_features = variational_dropout(output_features)
        concat_features = tf.concat([noisy_features] + seq_features, axis=-1)
      else:
        concat_features = tf.concat([output_features] + seq_features, axis=-1)
      regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=embedding_reg_lst)

      group_features = [cols_to_output_tensors[x] for x in group_columns] + \
                       [cols_to_output_tensors[x] for x in group_seq_columns]
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
