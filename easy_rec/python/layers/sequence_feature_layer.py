import logging
import os

import tensorflow as tf
from tensorflow.python.framework import ops

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers import seq_input_layer
from easy_rec.python.utils import conditional

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SequenceFeatureLayer(object):

  def __init__(self,
               feature_configs,
               feature_groups_config,
               ev_params=None,
               embedding_regularizer=None,
               kernel_regularizer=None,
               is_training=False,
               is_predicting=False):
    self._seq_feature_groups_config = []
    for x in feature_groups_config:
      for y in x.sequence_features:
        self._seq_feature_groups_config.append(y)
    self._seq_input_layer = None
    if len(self._seq_feature_groups_config) > 0:
      self._seq_input_layer = seq_input_layer.SeqInputLayer(
          feature_configs,
          self._seq_feature_groups_config,
          embedding_regularizer=embedding_regularizer,
          ev_params=ev_params)
    self._embedding_regularizer = embedding_regularizer
    self._kernel_regularizer = kernel_regularizer
    self._is_training = is_training
    self._is_predicting = is_predicting

  def negative_sampler_target_attention(self,
                                        dnn_config,
                                        deep_fea,
                                        concat_features,
                                        name,
                                        need_key_feature=True,
                                        allow_key_transform=False):
    cur_id, hist_id_col, seq_len, aux_hist_emb_list = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len'], deep_fea[
            'aux_hist_seq_emb_list']

    seq_max_len = tf.shape(hist_id_col)[1]
    seq_emb_dim = hist_id_col.shape[2]
    cur_id_dim = tf.shape(cur_id)[-1]
    batch_size = tf.shape(hist_id_col)[0]

    pos_feature = cur_id[:batch_size]
    neg_feature = cur_id[batch_size:]
    cur_id = tf.concat([
        pos_feature[:, tf.newaxis, :],
        tf.tile(neg_feature[tf.newaxis, :, :], multiples=[batch_size, 1, 1])
    ],
                       axis=1)  # noqa: E126
    neg_num_add_1 = tf.shape(cur_id)[1]
    hist_id_col_tmp = tf.tile(
        hist_id_col[:, :, :], multiples=[1, neg_num_add_1, 1])
    hist_id_col = tf.reshape(
        hist_id_col_tmp, [batch_size * neg_num_add_1, seq_max_len, seq_emb_dim])

    concat_features = tf.tile(
        concat_features[:, tf.newaxis, :], multiples=[1, neg_num_add_1, 1])
    seq_len = tf.tile(seq_len, multiples=[neg_num_add_1])

    if allow_key_transform and (cur_id_dim != seq_emb_dim):
      cur_id = tf.layers.dense(
          cur_id, seq_emb_dim, name='sequence_key_transform_layer')

    cur_ids = tf.tile(cur_id, [1, 1, seq_max_len])
    cur_ids = tf.reshape(
        cur_ids,
        tf.shape(hist_id_col))  # (B * neg_num_add_1, seq_max_len, seq_emb_dim)

    din_net = tf.concat(
        [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
        axis=-1)  # (B * neg_num_add_1, seq_max_len, seq_emb_dim*4)

    din_layer = dnn.DNN(
        dnn_config,
        self._kernel_regularizer,
        name,
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    din_net = din_layer(din_net)
    scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

    seq_len = tf.expand_dims(seq_len, 1)
    mask = tf.sequence_mask(seq_len)
    padding = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(mask, scores,
                      padding)  # [B*neg_num_add_1, 1, seq_max_len]

    # Scale
    scores = tf.nn.softmax(scores)  # (B * neg_num_add_1, 1, seq_max_len)
    hist_din_emb = tf.matmul(scores,
                             hist_id_col)  # [B * neg_num_add_1, 1, seq_emb_dim]
    hist_din_emb = tf.reshape(hist_din_emb,
                              [batch_size, neg_num_add_1, seq_emb_dim
                               ])  # [B * neg_num_add_1, seq_emb_dim]
    if len(aux_hist_emb_list) > 0:
      all_hist_dim_emb = [hist_din_emb]
      for hist_col in aux_hist_emb_list:
        cur_aux_hist = tf.matmul(scores, hist_col)
        outputs = tf.reshape(cur_aux_hist, [-1, seq_emb_dim])
        all_hist_dim_emb.append(outputs)
      hist_din_emb = tf.concat(all_hist_dim_emb, axis=1)
    if not need_key_feature:
      return hist_din_emb, concat_features
    din_output = tf.concat([hist_din_emb, cur_id], axis=2)
    return din_output, concat_features

  def target_attention(self,
                       dnn_config,
                       deep_fea,
                       name,
                       need_key_feature=True,
                       allow_key_transform=False,
                       transform_dnn=False):
    cur_id, hist_id_col, seq_len, aux_hist_emb_list = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len'], deep_fea[
            'aux_hist_seq_emb_list']

    seq_max_len = tf.shape(hist_id_col)[1]
    seq_emb_dim = hist_id_col.shape[2]
    cur_id_dim = cur_id.shape[-1]

    if allow_key_transform and (cur_id_dim != seq_emb_dim):
      if seq_emb_dim > cur_id_dim and not transform_dnn:
        cur_id = tf.pad(cur_id, [[0, 0], [0, seq_emb_dim - cur_id_dim]])
      else:
        cur_key_layer_name = 'sequence_key_transform_layer_' + name
        cur_id = tf.layers.dense(cur_id, seq_emb_dim, name=cur_key_layer_name)
        cur_fea_layer_name = 'sequence_fea_transform_layer_' + name
        hist_id_col = tf.layers.dense(
            hist_id_col, seq_emb_dim, name=cur_fea_layer_name)
    else:
      cur_id = cur_id[:tf.shape(hist_id_col)[0], ...]  # for negative sampler

    cur_ids = tf.tile(cur_id, [1, seq_max_len])
    cur_ids = tf.reshape(cur_ids,
                         tf.shape(hist_id_col))  # (B, seq_max_len, seq_emb_dim)

    din_net = tf.concat(
        [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
        axis=-1)  # (B, seq_max_len, seq_emb_dim*4)

    din_layer = dnn.DNN(
        dnn_config,
        self._kernel_regularizer,
        name,
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    din_net = din_layer(din_net)
    scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

    seq_len = tf.expand_dims(seq_len, 1)
    mask = tf.sequence_mask(seq_len)
    padding = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(mask, scores, padding)  # [B, 1, seq_max_len]

    # Scale
    scores = tf.nn.softmax(scores)  # (B, 1, seq_max_len)
    hist_din_emb = tf.matmul(scores, hist_id_col)  # [B, 1, seq_emb_dim]
    hist_din_emb = tf.reshape(hist_din_emb,
                              [-1, seq_emb_dim])  # [B, seq_emb_dim]
    if len(aux_hist_emb_list) > 0:
      all_hist_dim_emb = [hist_din_emb]
      for hist_col in aux_hist_emb_list:
        aux_hist_dim = hist_col.shape[-1]
        cur_aux_hist = tf.matmul(scores, hist_col)
        outputs = tf.reshape(cur_aux_hist, [-1, aux_hist_dim])
        all_hist_dim_emb.append(outputs)
      hist_din_emb = tf.concat(all_hist_dim_emb, axis=1)
    if not need_key_feature:
      return hist_din_emb
    din_output = tf.concat([hist_din_emb, cur_id], axis=1)
    return din_output

  def __call__(self,
               features,
               concat_features,
               all_seq_att_map_config,
               feature_name_to_output_tensors=None,
               negative_sampler=False,
               scope_name=None):
    logging.info('use sequence feature layer.')
    all_seq_fea = []
    # process all sequence features
    for seq_att_map_config in all_seq_att_map_config:
      group_name = seq_att_map_config.group_name
      allow_key_search = seq_att_map_config.allow_key_search
      need_key_feature = seq_att_map_config.need_key_feature
      allow_key_transform = seq_att_map_config.allow_key_transform
      transform_dnn = seq_att_map_config.transform_dnn

      place_on_cpu = os.getenv('place_embedding_on_cpu')
      place_on_cpu = eval(place_on_cpu) if place_on_cpu else False
      with conditional(self._is_predicting and place_on_cpu,
                       ops.device('/CPU:0')):
        seq_features = self._seq_input_layer(features, group_name,
                                             feature_name_to_output_tensors,
                                             allow_key_search, scope_name)

      # apply regularization for sequence feature key in seq_input_layer.

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
      if negative_sampler:
        seq_fea, concat_features = self.negative_sampler_target_attention(
            seq_dnn_config,
            seq_features,
            concat_features,
            name=cur_target_attention_name,
            need_key_feature=need_key_feature,
            allow_key_transform=allow_key_transform)
      else:
        seq_fea = self.target_attention(
            seq_dnn_config,
            seq_features,
            name=cur_target_attention_name,
            need_key_feature=need_key_feature,
            allow_key_transform=allow_key_transform,
            transform_dnn=transform_dnn)
      all_seq_fea.append(seq_fea)
    return concat_features, all_seq_fea
