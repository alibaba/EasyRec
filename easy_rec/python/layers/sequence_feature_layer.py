import logging

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import dnn
from easy_rec.python.layers import seq_input_layer

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SequenceFeatureLayer(object):

  def __init__(self,
               feature_configs,
               feature_groups_config,
               ev_params=None,
               embedding_regularizer=None,
               is_training=False):
    self._seq_feature_groups_config = []
    for x in feature_groups_config:
      for y in x.sequence_features:
        self._seq_feature_groups_config.append(y)
    self._seq_input_layer = None
    if len(self._seq_feature_groups_config) > 0:
      self._seq_input_layer = seq_input_layer.SeqInputLayer(
          feature_configs, self._seq_feature_groups_config, ev_params=ev_params)
    self._embedding_regularizer = embedding_regularizer
    self._is_training = is_training

  def target_attention(self,
                       dnn_config,
                       deep_fea,
                       name,
                       need_key_feature=True,
                       allow_key_transform=False):
    cur_id, hist_id_col, seq_len, aux_hist_emb_list = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len'], deep_fea[
            'aux_hist_seq_emb_list']

    seq_max_len = tf.shape(hist_id_col)[1]
    seq_emb_dim = hist_id_col.shape[2]
    cur_id_dim = tf.shape(cur_id)[-1]

    cur_id = cur_id[:tf.shape(hist_id_col)[0], ...]  # for negative sampler

    if allow_key_transform and (cur_id_dim != seq_emb_dim):
      cur_id = tf.layers.dense(
          cur_id, seq_emb_dim, name='sequence_key_transform_layer')

    cur_ids = tf.tile(cur_id, [1, seq_max_len])
    cur_ids = tf.reshape(cur_ids,
                         tf.shape(hist_id_col))  # (B, seq_max_len, seq_emb_dim)

    din_net = tf.concat(
        [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
        axis=-1)  # (B, seq_max_len, seq_emb_dim*4)

    din_layer = dnn.DNN(dnn_config, None, name, self._is_training)
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
        cur_aux_hist = tf.matmul(scores, hist_col)
        outputs = tf.reshape(cur_aux_hist, [-1, seq_emb_dim])
        all_hist_dim_emb.append(outputs)
      hist_din_emb = tf.concat(all_hist_dim_emb, axis=1)
    if not need_key_feature:
      return hist_din_emb
    din_output = tf.concat([hist_din_emb, cur_id], axis=1)
    return din_output

  def __call__(self,
               features,
               all_seq_att_map_config,
               feature_name_to_output_tensors=None):
    logging.info('use sequence feature layer.')
    all_seq_fea = []
    # process all sequence features
    for seq_att_map_config in all_seq_att_map_config:
      group_name = seq_att_map_config.group_name
      allow_key_search = seq_att_map_config.allow_key_search
      need_key_feature = seq_att_map_config.need_key_feature
      allow_key_transform = seq_att_map_config.allow_key_transform
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
          seq_dnn_config,
          seq_features,
          name=cur_target_attention_name,
          need_key_feature=need_key_feature,
          allow_key_transform=allow_key_transform)
      all_seq_fea.append(seq_fea)
    # concat all seq_fea
    all_seq_fea = tf.concat(all_seq_fea, axis=1)
    return all_seq_fea
