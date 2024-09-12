# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.match_model import MatchModel
from easy_rec.python.protos.dssm_pb2 import DSSM as DSSMConfig
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.simi_pb2 import Similarity
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses


class DSSM(MatchModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DSSM, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert self._model_config.WhichOneof('model') == 'dssm', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dssm
    assert isinstance(self._model_config, DSSMConfig)

    # copy_obj so that any modification will not affect original config
    self.user_tower = copy_obj(self._model_config.user_tower)
    self.user_tower_feature, _ = self._input_layer(self._feature_dict, 'user')
    # copy_obj so that any modification will not affect original config
    self.item_tower = copy_obj(self._model_config.item_tower)
    self.item_tower_feature, _ = self._input_layer(self._feature_dict, 'item')
    self._user_tower_emb = None
    self._item_tower_emb = None

  def build_predict_graph(self):
    num_user_dnn_layer = len(self.user_tower.dnn.hidden_units)
    last_user_hidden = self.user_tower.dnn.hidden_units.pop()
    user_dnn = dnn.DNN(self.user_tower.dnn, self._l2_reg, 'user_dnn',
                       self._is_training)
    user_tower_emb = user_dnn(self.user_tower_feature)
    user_tower_emb = tf.layers.dense(
        inputs=user_tower_emb,
        units=last_user_hidden,
        kernel_regularizer=self._l2_reg,
        name='user_dnn/dnn_%d' % (num_user_dnn_layer - 1))

    num_item_dnn_layer = len(self.item_tower.dnn.hidden_units)
    last_item_hidden = self.item_tower.dnn.hidden_units.pop()
    item_dnn = dnn.DNN(self.item_tower.dnn, self._l2_reg, 'item_dnn',
                       self._is_training)
    item_tower_emb = item_dnn(self.item_tower_feature)
    item_tower_emb = tf.layers.dense(
        inputs=item_tower_emb,
        units=last_item_hidden,
        kernel_regularizer=self._l2_reg,
        name='item_dnn/dnn_%d' % (num_item_dnn_layer - 1))

    if self._model_config.simi_func == Similarity.COSINE:
      user_tower_emb = self.norm(user_tower_emb)
      item_tower_emb = self.norm(item_tower_emb)
      temperature = self._model_config.temperature
    else:
      temperature = 1.0

    user_item_sim = self.sim(user_tower_emb, item_tower_emb) / temperature
    if self._model_config.scale_simi:
      sim_w = tf.get_variable(
          'sim_w',
          dtype=tf.float32,
          shape=(1),
          initializer=tf.ones_initializer())
      sim_b = tf.get_variable(
          'sim_b',
          dtype=tf.float32,
          shape=(1),
          initializer=tf.zeros_initializer())
      y_pred = user_item_sim * tf.abs(sim_w) + sim_b
    else:
      y_pred = user_item_sim

    if self._is_point_wise:
      y_pred = tf.reshape(y_pred, [-1])

    if self._loss_type == LossType.CLASSIFICATION:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      y_pred = self._mask_in_batch(y_pred)
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.softmax(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

    self._prediction_dict['user_tower_emb'] = user_tower_emb
    self._prediction_dict['item_tower_emb'] = item_tower_emb
    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.as_string(user_tower_emb), axis=-1, separator=',')
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_tower_emb), axis=-1, separator=',')
    return self._prediction_dict

  def get_outputs(self):
    if self._loss_type == LossType.CLASSIFICATION:
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_tower_emb',
          'item_tower_emb'
      ]
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      self._prediction_dict['logits'] = tf.squeeze(
          self._prediction_dict['logits'], axis=-1)
      self._prediction_dict['probs'] = tf.nn.sigmoid(
          self._prediction_dict['logits'])
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_tower_emb',
          'item_tower_emb'
      ]
    elif self._loss_type == LossType.L2_LOSS:
      return ['y', 'user_emb', 'item_emb', 'user_tower_emb', 'item_tower_emb']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))

  def build_output_dict(self):
    output_dict = super(DSSM, self).build_output_dict()
    output_dict['user_tower_feature'] = tf.reduce_join(
        tf.as_string(self.user_tower_feature), axis=-1, separator=',')
    output_dict['item_tower_feature'] = tf.reduce_join(
        tf.as_string(self.item_tower_feature), axis=-1, separator=',')
    return output_dict

  def build_rtp_output_dict(self):
    output_dict = super(DSSM, self).build_rtp_output_dict()
    if 'user_tower_emb' not in self._prediction_dict:
      raise ValueError(
          'User tower embedding does not exist. Please checking predict graph.')
    output_dict['user_embedding_output'] = tf.identity(
        self._prediction_dict['user_tower_emb'], name='user_embedding_output')
    if 'item_tower_emb' not in self._prediction_dict:
      raise ValueError(
          'Item tower embedding does not exist. Please checking predict graph.')
    output_dict['item_embedding_output'] = tf.identity(
        self._prediction_dict['item_tower_emb'], name='item_embedding_output')
    if self._loss_type == LossType.CLASSIFICATION:
      if 'probs' not in self._prediction_dict:
        raise ValueError(
            'Probs output does not exist. Please checking predict graph.')
      output_dict['rank_predict'] = tf.identity(
          self._prediction_dict['probs'], name='rank_predict')
    return output_dict
