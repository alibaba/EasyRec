# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.builders import loss_builder
from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.simi_pb2 import Similarity

from easy_rec.python.protos.rocket_launching_pb2 import RocketLaunching as RocketLaunchingConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class RocketLaunching(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(RocketLaunching, self).__init__(model_config, feature_configs,
                                          features, labels, is_training)
    assert self._model_config.WhichOneof('model') == 'rocket_launching', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.rocket_launching
    assert isinstance(self._model_config, RocketLaunchingConfig)
    if self._labels is not None:
      self._label_name = list(self._labels.keys())[0]

    self._features, _ = self._input_layer(self._feature_dict, 'all')

  def sim(self, feature_emb1, feature_emb2):
    emb1_emb2_sim = tf.reduce_sum(
        tf.multiply(feature_emb1, feature_emb2), axis=1, keepdims=True)
    return emb1_emb2_sim

  def norm(self, fea):
    fea_norm = tf.nn.l2_normalize(fea, axis=1)
    return fea_norm

  def feature_based_sim(self, feature_based_distillation, i, j):
    booster_feature_no_gradient = tf.stop_gradient(
        self.booster_feature['hidden_layer' + str(j)])
    if feature_based_distillation == Similarity.COSINE:
      booster_feature_no_gradient_norm = self.norm(booster_feature_no_gradient)
      light_feature_norm = self.norm(self.light_feature['hidden_layer' +
                                                        str(i)])
      sim_middle_layer = tf.reduce_mean(
          self.sim(booster_feature_no_gradient_norm, light_feature_norm))
      return sim_middle_layer
    else:
      return tf.sqrt(
          tf.reduce_sum(
              tf.square(booster_feature_no_gradient -
                        self.light_feature['hidden_layer' + str(i)])))

  def build_predict_graph(self):
    self.hidden_layer_feature_output = self._model_config.feature_based_distillation
    
    features = self._features
    if self._model_config.use_sequence_encoder:
      seq_encoding = self.get_sequence_encoding(is_training=self._is_training)
      if seq_encoding is not None:
        features = tf.concat([features, seq_encoding], axis=-1)

    if self._model_config.HasField('share_dnn'):
      share_dnn_layer = dnn.DNN(self._model_config.share_dnn, self._l2_reg,
                                'share_dnn', self._is_training)
      features = share_dnn_layer(features)

    booster_dnn_layer = dnn.DNN(self._model_config.booster_dnn, self._l2_reg,
                                'booster_dnn', self._is_training)
    light_dnn_layer = dnn.DNN(self._model_config.light_dnn, self._l2_reg,
                              'light_dnn', self._is_training)
    self.booster_feature = booster_dnn_layer(features,
                                              self.hidden_layer_feature_output)
    input_embedding_stop_gradient = tf.stop_gradient(features)
    self.light_feature = light_dnn_layer(input_embedding_stop_gradient,
                                          self.hidden_layer_feature_output)

    if self._model_config.feature_based_distillation:
      booster_out = tf.layers.dense(
          self.booster_feature['hidden_layer_end'],
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='booster_output')

      light_out = tf.layers.dense(
          self.light_feature['hidden_layer_end'],
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='light_output')
    else:
      booster_out = tf.layers.dense(
          self.booster_feature,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='booster_output')

      light_out = tf.layers.dense(
          self.light_feature,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='light_output')
    
    if len(self._losses) == 0:
      self._prediction_dict.update(
          self._output_to_prediction_impl(
              booster_out, 
              loss_type=self._loss_type, 
              num_class=self._num_class,
              suffix='_booster'))
      self._prediction_dict.update(
          self._output_to_prediction_impl(
              light_out, 
              loss_type=self._loss_type, 
              num_class=self._num_class,
              suffix='_light'))
    else:
      for loss in self._losses:
        loss_param = loss.WhichOneof('loss_param')
        if loss_param is not None:
          loss_param = getattr(loss, loss_param)
        self._prediction_dict.update(
            self._output_to_prediction_impl(
                booster_out,
                loss_type=loss.loss_type,
                num_class=self._num_class,
                loss_param=loss_param,
                suffix='_booster'))
        self._prediction_dict.update(
            self._output_to_prediction_impl(
                light_out,
                loss_type=loss.loss_type,
                num_class=self._num_class,
                loss_param=loss_param,
                suffix='_light'))

    return self._prediction_dict

  def build_loss_graph(self):
    logits_booster = self._prediction_dict['logits_booster']
    logits_light = self._prediction_dict['logits_light']
    self.feature_distillation_function = self._model_config.feature_distillation_function

    # feature_based_distillation loss
    if self._model_config.feature_based_distillation:
      booster_hidden_units = self._model_config.booster_dnn.hidden_units
      light_hidden_units = self._model_config.light_dnn.hidden_units
      count = 0

      for i, unit_i in enumerate(light_hidden_units):
        for j, unit_j in enumerate(booster_hidden_units):
          if light_hidden_units[i] == booster_hidden_units[j]:
            self._prediction_dict[
                'similarity_' + str(count)] = self.feature_based_sim(
                    self._model_config.feature_based_distillation, i, j)
            count += 1
            break

    if len(self._losses) == 0:
      self._loss_dict.update(
          self._build_loss_impl(
              self._loss_type,
              label_name=self._label_name,
              loss_weight=self._sample_weight,
              num_class=self._num_class,
              suffix='_booster'))
      self._loss_dict.update(
          self._build_loss_impl(
              self._loss_type,
              label_name=self._label_name,
              loss_weight=self._sample_weight,
              num_class=self._num_class,
              suffix='_light'))
    else:
      for loss in self._losses:
        loss_param = loss.WhichOneof('loss_param')
        if loss_param is not None:
          loss_param = getattr(loss, loss_param)
        booster_loss_ops = self._build_loss_impl(
            loss.loss_type,
            label_name=self._label_name,
            loss_weight=self._sample_weight,
            num_class=self._num_class,
            suffix='_booster',
            loss_name=loss.loss_name,
            loss_param=loss_param)
        light_loss_ops = self._build_loss_impl(
            loss.loss_type,
            label_name=self._label_name,
            loss_weight=self._sample_weight,
            num_class=self._num_class,
            suffix='_light',
            loss_name=loss.loss_name,
            loss_param=loss_param)
        for loss_name, loss_value in booster_loss_ops.items():
            self._loss_dict[loss_name] = loss_value * loss.weight
        for loss_name, loss_value in light_loss_ops.items():
            self._loss_dict[loss_name] = loss_value * loss.weight

    booster_logits_no_grad = tf.stop_gradient(logits_booster)

    sample_weight = self._sample_weight
    if self._num_class > 1:
      sample_weight = self._sample_weight[:, tf.newaxis]
    self._loss_dict['hint_loss'] = loss_builder.build(
        LossType.L2_LOSS,
        label=booster_logits_no_grad,
        pred=logits_light,
        loss_weight=sample_weight)

    if self._model_config.feature_based_distillation:
      for key, value in self._prediction_dict.items():
        if key.startswith('similarity_'):
          self._loss_dict[key] = -0.1 * value
      return self._loss_dict
    else:
      return self._loss_dict

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    loss_types = {self._loss_type}
    if len(self._losses) > 0:
      loss_types = {loss.loss_type for loss in self._losses}
    for metric in eval_config.metrics_set:
      metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=loss_types,
              label_name=self._label_name,
              num_class=self._num_class,
              suffix='_light'))
      metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=loss_types,
              label_name=self._label_name,
              num_class=self._num_class,
              suffix='_booster'))
    return metric_dict

  def get_outputs(self):
    outputs = []
    outputs.extend(
        self._get_outputs_impl(
            self._loss_type, self._num_class, suffix='_light'))
    outputs.extend(
        self._get_outputs_impl(
            self._loss_type, self._num_class, suffix='_booster'))
    return outputs
