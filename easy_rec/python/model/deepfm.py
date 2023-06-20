# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import fm
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos.deepfm_pb2 import DeepFM as DeepFMConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DeepFM(RankModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DeepFM, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'deepfm', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.deepfm
    assert isinstance(self._model_config, DeepFMConfig)

    # backward compatibility
    if self._model_config.HasField('wide_regularization'):
      tf.logging.warn(
          'wide_regularization is deprecated, please use l2_regularization')

    self._wide_features, _ = self._input_layer(self._feature_dict, 'wide')
    self._deep_features, self._fm_features = self._input_layer(
        self._feature_dict, 'deep')
    if 'fm' in self._input_layer._feature_groups:
      _, self._fm_features = self._input_layer(self._feature_dict, 'fm')

  def build_input_layer(self, model_config, feature_configs):
    # overwrite create input_layer to support wide_output_dim
    has_final = len(model_config.deepfm.final_dnn.hidden_units) > 0
    if not has_final:
      assert model_config.deepfm.wide_output_dim == model_config.num_class
    self._wide_output_dim = model_config.deepfm.wide_output_dim
    super(DeepFM, self).build_input_layer(model_config, feature_configs)

  def build_predict_graph(self):
    # Wide
    wide_fea = tf.reduce_sum(
        self._wide_features, axis=1, keepdims=True, name='wide_feature')

    # FM
    fm_fea = fm.FM(name='fm_feature')(self._fm_features)
    self._fm_outputs = fm_fea

    # Deep
    deep_layer = dnn.DNN(self._model_config.dnn, self._l2_reg, 'deep_feature',
                         self._is_training)
    deep_fea = deep_layer(self._deep_features)

    # Final
    if len(self._model_config.final_dnn.hidden_units) > 0:
      all_fea = tf.concat([wide_fea, fm_fea, deep_fea], axis=1)
      final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                                'final_dnn', self._is_training)
      all_fea = final_dnn_layer(all_fea)
      output = tf.layers.dense(
          all_fea,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='output')
    else:
      if self._num_class > 1:
        fm_fea = tf.layers.dense(
            fm_fea,
            self._num_class,
            kernel_regularizer=self._l2_reg,
            name='fm_logits')
      else:
        fm_fea = tf.reduce_sum(fm_fea, 1, keepdims=True)
      deep_fea = tf.layers.dense(
          deep_fea,
          self._num_class,
          kernel_regularizer=self._l2_reg,
          name='deep_logits')
      output = wide_fea + fm_fea + deep_fea

    self._add_to_prediction_dict(output)

    return self._prediction_dict

  def build_feature_output_dict(self):
    outputs = super(DeepFM, self).build_feature_output_dict()
    outputs.update({
        'wide_features':
            tf.reduce_join(
                tf.as_string(self._wide_features), axis=-1, separator=','),
        'deep_features':
            tf.reduce_join(
                tf.as_string(self._deep_features), axis=-1, separator=','),
        'fm_outputs':
            tf.reduce_join(
                tf.as_string(self._fm_outputs), axis=-1, separator=',')
    })
    return outputs
