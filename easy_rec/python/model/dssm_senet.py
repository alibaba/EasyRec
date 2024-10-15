# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import senet
from easy_rec.python.model.dssm import DSSM
from easy_rec.python.model.match_model import MatchModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.simi_pb2 import Similarity
from easy_rec.python.utils.proto_util import copy_obj

from easy_rec.python.protos.dssm_senet_pb2 import DSSM_SENet as DSSM_SENet_Config  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses


class DSSM_SENet(DSSM):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):

    MatchModel.__init__(self, model_config, feature_configs, features, labels,
                        is_training)

    assert self._model_config.WhichOneof('model') == 'dssm_senet', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dssm_senet
    assert isinstance(self._model_config, DSSM_SENet_Config)

    # copy_obj so that any modification will not affect original config
    self.user_tower = copy_obj(self._model_config.user_tower)

    self.user_seq_features, self.user_plain_features, self.user_feature_list = self._input_layer(
        self._feature_dict, 'user', is_combine=False)
    self.user_num_fields = len(self.user_feature_list)

    # copy_obj so that any modification will not affect original config
    self.item_tower = copy_obj(self._model_config.item_tower)

    self.item_seq_features, self.item_plain_features, self.item_feature_list = self._input_layer(
        self._feature_dict, 'item', is_combine=False)
    self.item_num_fields = len(self.item_feature_list)

    self._user_tower_emb = None
    self._item_tower_emb = None

  def build_predict_graph(self):
    user_senet = senet.SENet(
        num_fields=self.user_num_fields,
        num_squeeze_group=self.user_tower.senet.num_squeeze_group,
        reduction_ratio=self.user_tower.senet.reduction_ratio,
        l2_reg=self._l2_reg,
        name='user_senet')
    user_senet_output_list = user_senet(self.user_feature_list)
    user_senet_output = tf.concat(user_senet_output_list, axis=-1)

    num_user_dnn_layer = len(self.user_tower.dnn.hidden_units)
    last_user_hidden = self.user_tower.dnn.hidden_units.pop()
    user_dnn = dnn.DNN(self.user_tower.dnn, self._l2_reg, 'user_dnn',
                       self._is_training)
    user_tower_emb = user_dnn(user_senet_output)
    user_tower_emb = tf.layers.dense(
        inputs=user_tower_emb,
        units=last_user_hidden,
        kernel_regularizer=self._l2_reg,
        name='user_dnn/dnn_%d' % (num_user_dnn_layer - 1))

    item_senet = senet.SENet(
        num_fields=self.item_num_fields,
        num_squeeze_group=self.item_tower.senet.num_squeeze_group,
        reduction_ratio=self.item_tower.senet.reduction_ratio,
        l2_reg=self._l2_reg,
        name='item_senet')

    item_senet_output_list = item_senet(self.item_feature_list)
    item_senet_output = tf.concat(item_senet_output_list, axis=-1)

    num_item_dnn_layer = len(self.item_tower.dnn.hidden_units)
    last_item_hidden = self.item_tower.dnn.hidden_units.pop()
    item_dnn = dnn.DNN(self.item_tower.dnn, self._l2_reg, 'item_dnn',
                       self._is_training)
    item_tower_emb = item_dnn(item_senet_output)
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

  def build_output_dict(self):
    output_dict = MatchModel.build_output_dict(self)

    return output_dict
