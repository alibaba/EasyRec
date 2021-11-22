# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.feature_column.feature_group import FeatureGroup
from easy_rec.python.layers import variational_dropout_layer
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep
from easy_rec.python.layers.common_layers import text_cnn
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
               embedding_regularizer=None,
               kernel_regularizer=None,
               is_training=False):
    self._feature_groups = {
        x.group_name: FeatureGroup(x) for x in feature_groups_config
    }
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
          elif sequence_combiner.WhichOneof('combiner') == 'text_cnn':
            num_filters = sequence_combiner.text_cnn.num_filters
            filter_sizes = sequence_combiner.text_cnn.filter_sizes
            cnn_feature = text_cnn(seq_feature, filter_sizes, num_filters)
            seq_features.append(cnn_feature)
            cols_to_output_tensors[column] = cnn_feature
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
