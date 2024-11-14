# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.model.match_model import MatchModel
from easy_rec.python.protos.dat_pb2 import DAT as DATConfig
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.utils.proto_util import copy_obj

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DAT(MatchModel):
  """Dual Augmented Two-tower Model."""

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DAT, self).__init__(model_config, feature_configs, features, labels,
                              is_training)
    assert self._model_config.WhichOneof('model') == 'dat', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')

    feature_group_names = [
        fg.group_name for fg in self._model_config.feature_groups
    ]
    assert 'user' in feature_group_names, 'user feature group not found'
    assert 'item' in feature_group_names, 'item feature group not found'
    assert 'user_id_augment' in feature_group_names, 'user_id_augment feature group not found'
    assert 'item_id_augment' in feature_group_names, 'item_id_augment feature group not found'

    self._model_config = self._model_config.dat
    assert isinstance(self._model_config, DATConfig)

    self.user_tower = copy_obj(self._model_config.user_tower)
    self.user_deep_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.user_augmented_vec, _ = self._input_layer(self._feature_dict,
                                                   'user_id_augment')

    self.item_tower = copy_obj(self._model_config.item_tower)
    self.item_deep_feature, _ = self._input_layer(self._feature_dict, 'item')
    self.item_augmented_vec, _ = self._input_layer(self._feature_dict,
                                                   'item_id_augment')

    self._user_tower_emb = None
    self._item_tower_emb = None

  def build_predict_graph(self):
    num_user_dnn_layer = len(self.user_tower.dnn.hidden_units)
    last_user_hidden = self.user_tower.dnn.hidden_units.pop()
    user_dnn = dnn.DNN(self.user_tower.dnn, self._l2_reg, 'user_dnn',
                       self._is_training)

    user_tower_feature = tf.concat(
        [self.user_deep_feature, self.user_augmented_vec], axis=-1)
    user_tower_emb = user_dnn(user_tower_feature)
    user_tower_emb = tf.layers.dense(
        inputs=user_tower_emb,
        units=last_user_hidden,
        kernel_regularizer=self._l2_reg,
        name='user_dnn/dnn_%d' % (num_user_dnn_layer - 1))

    num_item_dnn_layer = len(self.item_tower.dnn.hidden_units)
    last_item_hidden = self.item_tower.dnn.hidden_units.pop()
    item_dnn = dnn.DNN(self.item_tower.dnn, self._l2_reg, 'item_dnn',
                       self._is_training)

    item_tower_feature = tf.concat(
        [self.item_deep_feature, self.item_augmented_vec], axis=-1)
    item_tower_emb = item_dnn(item_tower_feature)
    item_tower_emb = tf.layers.dense(
        inputs=item_tower_emb,
        units=last_item_hidden,
        kernel_regularizer=self._l2_reg,
        name='item_dnn/dnn_%d' % (num_item_dnn_layer - 1))

    user_tower_emb = self.norm(user_tower_emb)
    item_tower_emb = self.norm(item_tower_emb)
    temperature = self._model_config.temperature

    y_pred = self.sim(user_tower_emb, item_tower_emb) / temperature

    if self._is_point_wise:
      raise ValueError('Currently DAT model only supports list wise mode.')

    if self._loss_type == LossType.CLASSIFICATION:
      raise ValueError(
          'Currently DAT model only supports SOFTMAX_CROSS_ENTROPY loss.')
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

    augmented_p_u = tf.stop_gradient(user_tower_emb)
    augmented_p_i = tf.stop_gradient(item_tower_emb)

    self._prediction_dict['augmented_p_u'] = augmented_p_u
    self._prediction_dict['augmented_p_i'] = augmented_p_i

    self._prediction_dict['augmented_a_u'] = self.user_augmented_vec
    self._prediction_dict['augmented_a_i'] = self.item_augmented_vec

    return self._prediction_dict

  def get_outputs(self):
    if self._loss_type == LossType.CLASSIFICATION:
      raise ValueError(
          'Currently DAT model only supports SOFTMAX_CROSS_ENTROPY loss.')
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      self._prediction_dict['logits'] = tf.squeeze(
          self._prediction_dict['logits'], axis=-1)
      self._prediction_dict['probs'] = tf.nn.sigmoid(
          self._prediction_dict['logits'])
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_tower_emb',
          'item_tower_emb', 'augmented_p_u', 'augmented_p_i', 'augmented_a_u',
          'augmented_a_i'
      ]
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))

  def build_output_dict(self):
    output_dict = super(DAT, self).build_output_dict()
    return output_dict
