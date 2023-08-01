# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.match_model import MatchModel
from easy_rec.python.protos.simi_pb2 import Similarity

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses
metrics = tf.metrics


class PDN(MatchModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(PDN, self).__init__(model_config, feature_configs, features, labels,
                              is_training)
    assert self._model_config.WhichOneof('model') == 'pdn', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.pdn

    self._user_features, _ = self._input_layer(self._feature_dict, 'user')
    self._item_features, _ = self._input_layer(self._feature_dict, 'item')

    if self._input_layer.has_group('bias'):
      self._bias_features, _ = self._input_layer(self._feature_dict, 'bias')
    else:
      self._bias_features = None

    self._u2i_seq, self._seq_len = self._get_seq_features('u2i_seq')
    self._i_seq, _ = self._get_seq_features('i_seq')
    self._i2i_seq, _ = self._get_seq_features('i2i_seq')

  def build_predict_graph(self):
    trigger_out = self._build_trigger_net()
    sim_out = self._build_similarity_net()
    logits = tf.multiply(sim_out, trigger_out)

    seq_mask = tf.to_float(
        tf.sequence_mask(self._seq_len,
                         tf.shape(sim_out)[1]))
    logits = tf.reduce_sum(logits * seq_mask[:, :, None], axis=1)

    direct_logits = self._build_direct_net()
    if direct_logits is not None:
      logits += direct_logits

    bias_logits = self._build_bias_net()
    if bias_logits is not None:
      logits += bias_logits

    logits = tf.squeeze(logits, axis=1)
    probs = 1 - tf.exp(-logits)  # map [0, inf) to [0, 1)

    self._prediction_dict['probs'] = probs
    self._prediction_dict['logits'] = tf.log(
        tf.clip_by_value(probs, 1e-8, 1 - 1e-8))
    return self._prediction_dict

  def _get_seq_features(self, name):
    seqs, _, _ = self._input_layer(self._feature_dict, name, is_combine=False)
    seq_len = seqs[0][1]
    seq = tf.concat([x[0] for x in seqs], axis=2)
    return seq, seq_len

  def _build_trigger_net(self):
    user_dnn_layer = dnn.DNN(self._model_config.user_dnn, self._l2_reg,
                             'user_dnn', self._is_training)
    user_fea = user_dnn_layer(self._user_features)

    trigger_seq = tf.concat([self._u2i_seq, self._i_seq], axis=2)
    u2i_dnn_layer = dnn.DNN(self._model_config.u2i_dnn, self._l2_reg, 'u2i_dnn',
                            self._is_training)
    trigger_seq_fea = u2i_dnn_layer(trigger_seq)

    trigger_merge_fea = trigger_seq_fea + user_fea[:, None, :]
    trigger_dnn_layer = dnn.DNN(
        self._model_config.trigger_dnn,
        self._l2_reg,
        'trigger_dnn',
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)

    # output: N x seq_len x d, d is usually set to 1
    trigger_out = trigger_dnn_layer(trigger_merge_fea)
    # exp(x): map (-inf, inf) to (0, inf)
    trigger_out = tf.exp(trigger_out)

    self._prediction_dict['trigger_out'] = tf.reduce_join(
        tf.reduce_join(
            tf.as_string(trigger_out, precision=4, shortest=True),
            axis=2,
            separator=','),
        axis=1,
        separator=';')
    return trigger_out

  def _build_similarity_net(self):
    item_dnn_layer = dnn.DNN(self._model_config.item_dnn, self._l2_reg,
                             'item_dnn', self._is_training)
    item_fea = item_dnn_layer(self._item_features)

    sim_side_dnn_layer = dnn.DNN(self._model_config.i2i_dnn, self._l2_reg,
                                 'i2i_dnn', self._is_training)
    sim_seq_fea = sim_side_dnn_layer(self._i_seq)

    sim_seq_cross = sim_seq_fea * item_fea[:, None, :]

    item_fea_tile = tf.tile(item_fea[:, None, :],
                            [1, tf.shape(sim_seq_fea)[1], 1])

    sim_seq_concat = tf.concat(
        [sim_seq_cross, sim_seq_cross, self._i2i_seq, item_fea_tile], axis=2)
    sim_dnn_layer = dnn.DNN(
        self._model_config.sim_dnn,
        self._l2_reg,
        'sim_dnn',
        self._is_training,
        last_layer_no_activation=True,
        last_layer_no_batch_norm=True)
    # output: N x seq_len x 1
    sim_out = sim_dnn_layer(sim_seq_concat)
    # exp(x): map (-inf, inf) to (0, inf)
    sim_out = tf.exp(sim_out)

    self._prediction_dict['sim_out'] = tf.reduce_join(
        tf.reduce_join(
            tf.as_string(sim_out, precision=4, shortest=True),
            axis=2,
            separator=','),
        axis=1,
        separator=';')
    return sim_out

  def _build_direct_net(self):
    if self._model_config.HasField('direct_user_dnn') and \
       self._model_config.HasField('direct_item_dnn'):
      direct_user_layer = dnn.DNN(
          self._model_config.direct_user_dnn,
          'direct_user_dnn',
          self._is_training,
          last_layer_no_activation=True,
          last_layer_no_batch_norm=True)
      direct_user_out = direct_user_layer(self._user_features)
      direct_item_layer = dnn.DNN(
          self._model_config.direct_item_dnn,
          'direct_item_dnn',
          self._is_training,
          last_layer_no_activation=True,
          last_layer_no_batch_norm=True)
      direct_item_out = direct_item_layer(self._item_features)

      if self._model_config.simi_func == Similarity.COSINE:
        direct_user_out = self.norm(direct_user_out)
        direct_item_out = self.norm(direct_item_out)

      self._prediction_dict['direct_user_embedding'] = direct_user_out
      self._prediction_dict['direct_item_embedding'] = direct_item_out
      direct_logits = tf.reduce_sum(direct_user_out * direct_item_out, axis=1)

      if self._model_config.scale_simi:
        sim_w = tf.get_variable(
            'direct_net/sim_w',
            dtype=tf.float32,
            shape=(1),
            initializer=tf.ones_initializer())
        sim_b = tf.get_variable(
            'direct_net/sim_b',
            dtype=tf.float32,
            shape=(1),
            initializer=tf.zeros_initializer())
        direct_logits = direct_logits * tf.abs(sim_w) + sim_b

      return tf.nn.softplus(direct_logits)
    else:
      return None

  def _build_bias_net(self):
    if self._model_config.HasField('bias_dnn'):
      assert self._bias_features is not None, 'bias group must be defined'
      bias_dnn_layer = dnn.DNN(
          self._model_config.bias_dnn,
          self._l2_reg,
          'bias_dnn',
          self._is_training,
          last_layer_no_activation=True,
          last_layer_no_batch_norm=True)
      bias_logits = bias_dnn_layer(self._bias_features)
      return tf.nn.softplus(bias_logits)
    else:
      return None

  def get_outputs(self):
    return ['logits', 'probs', 'trigger_out', 'sim_out']
