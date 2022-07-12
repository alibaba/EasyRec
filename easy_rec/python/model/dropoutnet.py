# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.loss.pairwise_loss import pairwise_loss
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.utils.proto_util import copy_obj
from easy_rec.python.utils.shape_utils import get_shape_list

from easy_rec.python.protos.dropoutnet_pb2 import DropoutNet as DropoutNetConfig  # NOQA
from easy_rec.python.loss.softmax_loss_with_negative_mining import softmax_loss_with_negative_mining  # NOQA
from easy_rec.python.protos.dropoutnet_pb2 import DropoutNet as DropoutNetConfig  # NOQA
if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses


def cosine_similarity(user_emb, item_emb):
  user_item_sim = tf.reduce_sum(
      tf.multiply(user_emb, item_emb), axis=1, name='cosine')
  return user_item_sim


class DropoutNet(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DropoutNet, self).__init__(model_config, feature_configs, features,
                                     labels, is_training)
    self._losses = self._model_config.losses
    assert self._model_config.WhichOneof(
        'model'
    ) == 'dropoutnet', 'invalid model config: %s' % self._model_config.WhichOneof(
        'model')
    self._model_config = self._model_config.dropoutnet
    assert isinstance(self._model_config, DropoutNetConfig)

    # copy_obj so that any modification will not affect original config
    self.user_content_layers = copy_obj(self._model_config.user_content)
    self.user_preference_layers = copy_obj(self._model_config.user_preference)
    self.user_tower_layers = copy_obj(self._model_config.user_tower)
    self.user_content_feature, self.user_preference_feature = None, None
    if self._input_layer.has_group('user_content'):
      self.user_content_feature, _ = self._input_layer(self._feature_dict,
                                                       'user_content')
    if self._input_layer.has_group('user_preference'):
      self.user_preference_feature, _ = self._input_layer(
          self._feature_dict, 'user_preference')
    assert self.user_content_feature is not None or self.user_preference_feature is not None, 'no user feature'

    # copy_obj so that any modification will not affect original config
    self.item_content_layers = copy_obj(self._model_config.item_content)
    self.item_preference_layers = copy_obj(self._model_config.item_preference)
    self.item_tower_layers = copy_obj(self._model_config.item_tower)
    self.item_content_feature, self.item_preference_feature = None, None
    if self._input_layer.has_group('item_content'):
      self.item_content_feature, _ = self._input_layer(self._feature_dict,
                                                       'item_content')
    if self._input_layer.has_group('item_preference'):
      self.item_preference_feature, _ = self._input_layer(
          self._feature_dict, 'item_preference')
    assert self.item_content_feature is not None or self.item_preference_feature is not None, 'no item feature'

  def build_predict_graph(self):
    batch_size = get_shape_list(self.item_content_feature)[0]

    num_user_dnn_layer = len(self.user_tower_layers.hidden_units)
    last_user_hidden = self.user_tower_layers.hidden_units.pop()
    num_item_dnn_layer = len(self.item_tower_layers.hidden_units)
    last_item_hidden = self.item_tower_layers.hidden_units.pop()
    assert last_item_hidden == last_user_hidden, 'the last hidden layer size of user tower and item tower must be equal'

    # --------------------------build user tower-----------------------------------
    with tf.name_scope('user_tower'):
      user_features = []
      if self.user_content_feature is not None:
        user_content_dnn = dnn.DNN(self.user_content_layers, self._l2_reg,
                                   'user_content', self._is_training)
        content_feature = user_content_dnn(self.user_content_feature)
        user_features.append(content_feature)
      if self.user_preference_feature is not None:
        if self._is_training:
          prob = tf.random.uniform([batch_size])
          user_prefer_feature = tf.where(
              tf.less(prob, self._model_config.user_dropout_rate),
              tf.zeros_like(self.user_preference_feature),
              self.user_preference_feature)
        else:
          user_prefer_feature = self.user_preference_feature

        user_prefer_dnn = dnn.DNN(self.user_preference_layers, self._l2_reg,
                                  'user_preference', self._is_training)
        prefer_feature = user_prefer_dnn(user_prefer_feature)
        user_features.append(prefer_feature)

      user_tower_feature = tf.concat(user_features, axis=-1)

      user_dnn = dnn.DNN(self.user_tower_layers, self._l2_reg, 'user_dnn',
                         self._is_training)
      user_hidden = user_dnn(user_tower_feature)
      user_tower_emb = tf.layers.dense(
          inputs=user_hidden,
          units=last_user_hidden,
          kernel_regularizer=self._l2_reg,
          name='user_dnn/dnn_%d' % (num_user_dnn_layer - 1))

    # --------------------------build item tower-----------------------------------
    with tf.name_scope('item_tower'):
      item_features = []
      if self.item_content_feature is not None:
        item_content_dnn = dnn.DNN(self.item_content_layers, self._l2_reg,
                                   'item_content', self._is_training)
        content_feature = item_content_dnn(self.item_content_feature)
        item_features.append(content_feature)
      if self.item_preference_feature is not None:
        if self._is_training:
          prob = tf.random.uniform([batch_size])
          item_prefer_feature = tf.where(
              tf.less(prob, self._model_config.item_dropout_rate),
              tf.zeros_like(self.item_preference_feature),
              self.item_preference_feature)
        else:
          item_prefer_feature = self.item_preference_feature

        item_prefer_dnn = dnn.DNN(self.item_preference_layers, self._l2_reg,
                                  'item_preference', self._is_training)
        prefer_feature = item_prefer_dnn(item_prefer_feature)
        item_features.append(prefer_feature)

      item_tower_feature = tf.concat(item_features, axis=-1)

      item_dnn = dnn.DNN(self.item_tower_layers, self._l2_reg, 'item_dnn',
                         self._is_training)
      item_hidden = item_dnn(item_tower_feature)
      item_tower_emb = tf.layers.dense(
          inputs=item_hidden,
          units=last_item_hidden,
          kernel_regularizer=self._l2_reg,
          name='item_dnn/dnn_%d' % (num_item_dnn_layer - 1))

    user_emb = tf.nn.l2_normalize(user_tower_emb, axis=-1)
    item_emb = tf.nn.l2_normalize(item_tower_emb, axis=-1)
    cosine = cosine_similarity(user_emb, item_emb)
    self._prediction_dict['similarity'] = cosine
    self._prediction_dict['float_user_emb'] = user_emb
    self._prediction_dict['float_item_emb'] = item_emb
    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.as_string(user_emb), axis=-1, separator=',')
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_emb), axis=-1, separator=',')
    return self._prediction_dict

  def build_loss_graph(self):
    labels = list(self._labels.values())[0]
    logits = self._prediction_dict['similarity']
    for loss in self._losses:
      if loss.loss_type == LossType.SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING:
        assert self._model_config.HasField(
            'softmax_loss'), '`softmax_loss` must be configured'
        user_emb = self._prediction_dict['float_user_emb']
        item_emb = self._prediction_dict['float_item_emb']
        loss_value = softmax_loss_with_negative_mining(
            user_emb,
            item_emb,
            labels,
            self._model_config.softmax_loss.num_negative_samples,
            embed_normed=True,
            weights=self._sample_weight,
            margin=self._model_config.softmax_loss.margin,
            gamma=self._model_config.softmax_loss.gamma,
            t=self._model_config.softmax_loss.coefficient_of_support_vector)
        self._loss_dict['softmax_loss'] = loss_value * loss.weight
      elif loss.loss_type == LossType.PAIR_WISE_LOSS:
        loss_value = pairwise_loss(labels, logits)
        self._loss_dict['pairwise_loss'] = loss_value * loss.weight
      elif loss.loss_type == LossType.CLASSIFICATION:
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits,
                                                     self._sample_weight)
        self._loss_dict['sigmoid_loss'] = loss_value * loss.weight
    return self._loss_dict

  def build_metric_graph(self, eval_config):
    from easy_rec.python.core.easyrec_metrics import metrics_tf as metrics
    metric_dict = {}
    labels = list(self._labels.values())[0]
    sim_score = self._prediction_dict['similarity']
    prob = tf.nn.sigmoid(sim_score)
    predict = tf.greater(prob, 0.5)
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        metric_dict['auc'] = metrics.auc(
            labels, prob, weights=self._sample_weight)
      elif metric.WhichOneof('metric') == 'accuracy':
        metric_dict['accuracy'] = metrics.accuracy(
            tf.cast(labels, tf.bool), predict, weights=self._sample_weight)
      elif metric.WhichOneof('metric') == 'precision':
        metric_dict['precision'] = metrics.precision(
            labels, predict, weights=self._sample_weight)
      elif metric.WhichOneof('metric') == 'recall':
        metric_dict['recall'] = metrics.recall(
            labels, predict, weights=self._sample_weight)
      else:
        ValueError('invalid metric type: %s' % str(metric))
    return metric_dict

  def get_outputs(self):
    return ['similarity', 'user_emb', 'item_emb']
