# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import cmbf
from easy_rec.python.layers import dnn
from easy_rec.python.layers import layer_norm
from easy_rec.python.layers import mmoe
from easy_rec.python.layers import uniter
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.protos.dbmtl_pb2 import DBMTL as DBMTLConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class DBMTL(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DBMTL, self).__init__(model_config, feature_configs, features, labels,
                                is_training)
    assert self._model_config.WhichOneof('model') == 'dbmtl', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dbmtl
    assert isinstance(self._model_config, DBMTLConfig)

    if self._model_config.HasField('bottom_cmbf'):
      self._cmbf_layer = cmbf.CMBF(model_config, feature_configs, features,
                                   self._model_config.bottom_cmbf,
                                   self._input_layer)
    elif self._model_config.HasField('bottom_uniter'):
      self._uniter_layer = uniter.Uniter(model_config, feature_configs,
                                         features,
                                         self._model_config.bottom_uniter,
                                         self._input_layer)
    # elif self._model_config.HasField('sequence_dnn'):
    #   self._features, _, self._seq_features = self._input_layer(
    #       self._feature_dict, 'all', return_sequence=True)
    else:
      self._features, _ = self._input_layer(self._feature_dict, 'all')

    if self._input_layer.has_group('extra'):
      extra_features, _ = self._input_layer(self._feature_dict, 'extra')
      self._features = tf.concat([self._features, extra_features], axis=1)

    self._bias_features_dict = {}
    for task_tower_cfg in self._model_config.task_towers:
      for bias_tower_cfg in task_tower_cfg.bias_tower:
        if self._mode == tf.estimator.ModeKeys.TRAIN or bias_tower_cfg.infer_with_tower:
          if bias_tower_cfg.input not in self._bias_features_dict:
            self._bias_features_dict[
                bias_tower_cfg.input], _ = self._input_layer(
                    self._feature_dict, bias_tower_cfg.input)

    self._init_towers(self._model_config.task_towers)

  def build_predict_graph(self):
    if self._model_config.use_feature_bn:
      self._features = tf.layers.batch_normalization(
          self._features,
          training=self._is_training,
          trainable=True,
          name='feat/all/bn')
      for k, v in self._bias_features_dict.items():
        self._bias_features_dict[k] = tf.layers.batch_normalization(
            v,
            training=self._is_training,
            trainable=True,
            name='feat/%s/bn' % k)
    elif self._model_config.use_feature_ln:
      self._features = layer_norm.layer_norm(
          self._features, trainable=True, scope='feat/all/ln')
      for k, v in self._bias_features_dict.items():
        self._bias_features_dict[k] = layer_norm.layer_norm(
            v, trainable=True, name='feat/%s/ln' % k)

    if self._model_config.HasField('input_dropout_rate'):
      drop_rate = self._model_config.input_dropout_rate
      self._features = tf.layers.dropout(
          self._features,
          rate=drop_rate,
          training=self._is_training,
          name='input_dropout')

    if self._model_config.HasField('bottom_cmbf'):
      bottom_fea = self._cmbf_layer(self._is_training, l2_reg=self._l2_reg)
    elif self._model_config.HasField('bottom_uniter'):
      bottom_fea = self._uniter_layer(self._is_training, l2_reg=self._l2_reg)
    elif self._model_config.HasField('bottom_dnn'):
      bottom_dnn = dnn.DNN(
          self._model_config.bottom_dnn,
          self._l2_reg,
          name='bottom_dnn',
          is_training=self._is_training)
      bottom_fea = bottom_dnn(self._features)
    else:
      bottom_fea = self._features

    # if self._model_config.HasField('sequence_dnn'):
    #   sequence_dnn = dnn.DNN(
    #       self._model_config.sequence_dnn,
    #       self._l2_reg,
    #       name='sequence_dnn',
    #       is_training=self._is_training)
    #   sequence_fea = sequence_dnn(self._seq_features)
    #   bottom_fea = tf.concat([bottom_fea, sequence_fea], axis=-1)

    tf.summary.scalar('bottom_fea', tf.norm(bottom_fea))

    task_input_list = [bottom_fea] * self._task_num
    if self._model_config.use_sequence_encoder:
      seq_encoding = self.get_sequence_encoding(is_training=self._is_training)
      tf.summary.scalar('seq_norm', tf.norm(seq_encoding))
      if self._model_config.use_feature_ln and seq_encoding is not None:
        seq_encoding = layer_norm.layer_norm(
            seq_encoding, trainable=True, scope='feat/seq/ln')
      if seq_encoding is not None:
        if self._model_config.HasField('sequence_dnn'):
          if self._model_config.separate_dnn:
            for i in range(self._task_num):
              sequence_dnn = dnn.DNN(
                  self._model_config.sequence_dnn,
                  self._l2_reg,
                  name='sequence_dnn_%d' % i,
                  is_training=self._is_training)
              seq_fea = sequence_dnn(seq_encoding)
              task_input_list[i] = tf.concat([task_input_list[i], seq_fea],
                                             axis=-1)
          else:
            sequence_dnn = dnn.DNN(
                self._model_config.sequence_dnn,
                self._l2_reg,
                name='sequence_dnn',
                is_training=self._is_training)
            seq_fea = sequence_dnn(seq_encoding)
            task_input_list[i] = [tf.concat([bottom_fea, seq_fea], axis=-1)
                                  ] * self._task_num
        else:
          for i in range(self._task_num):
            task_input_list[i] = tf.concat([bottom_fea, seq_encoding], axis=-1)

    # MMOE block
    if self._model_config.HasField('expert_dnn'):
      assert not self._model_config.separate_dnn, 'mmoe cannot be used with separate_dnn'
      mmoe_layer = mmoe.MMOE(
          self._model_config.expert_dnn,
          l2_reg=self._l2_reg,
          num_task=self._task_num,
          num_expert=self._model_config.num_expert)
      task_input_list = mmoe_layer(task_input_list[0])

    tower_features = {}
    # task specify network
    for i, task_tower_cfg in enumerate(self._model_config.task_towers):
      tower_name = task_tower_cfg.tower_name
      if task_tower_cfg.HasField('dnn'):
        tower_dnn = dnn.DNN(
            task_tower_cfg.dnn,
            self._l2_reg,
            name=tower_name + '/dnn',
            is_training=self._is_training)
        tower_fea = tower_dnn(task_input_list[i])
        tower_features[tower_name] = tower_fea
      else:
        tower_features[tower_name] = task_input_list[i]

    tower_outputs = {}
    relation_features = {}
    # bayes network
    for task_tower_cfg in self._model_config.task_towers:
      tower_name = task_tower_cfg.tower_name
      relation_dnn = dnn.DNN(
          task_tower_cfg.relation_dnn,
          self._l2_reg,
          name=tower_name + '/relation_dnn',
          is_training=self._is_training)
      tower_inputs = [tower_features[tower_name]]
      for relation_tower_name in task_tower_cfg.relation_tower_names:
        tower_inputs.append(relation_features[relation_tower_name])
      relation_input = tf.concat(
          tower_inputs, axis=-1, name=tower_name + '/relation_input')
      relation_fea = relation_dnn(relation_input)
      relation_features[tower_name] = relation_fea

      output_logits = tf.layers.dense(
          relation_fea,
          task_tower_cfg.num_class,
          kernel_regularizer=self._l2_reg,
          name=tower_name + '/output')
      tf.summary.scalar(tower_name + '/output', tf.reduce_mean(output_logits))

      for bias_tower_cfg in task_tower_cfg.bias_tower:
        if self._mode == tf.estimator.ModeKeys.TRAIN or bias_tower_cfg.infer_with_tower:
          bias_dnn = dnn.DNN(
              task_tower_cfg.relation_dnn,
              self._l2_reg,
              name='%s/%s/bias_dnn' % (tower_name, bias_tower_cfg.input),
              is_training=self._is_training)
          bias_fea = bias_dnn(self._bias_features_dict[bias_tower_cfg.input])
          bias_logits = tf.layers.dense(
              bias_fea,
              task_tower_cfg.num_class,
              kernel_regularizer=self._l2_reg,
              name='%s/%s/bias_output' % (tower_name, bias_tower_cfg.input))
          if bias_tower_cfg.HasField(
              'bias_dropout_ratio') and bias_tower_cfg.bias_dropout_ratio > 0:
            bias_logits = tf.nn.dropout(
                bias_logits,
                keep_prob=1 - bias_tower_cfg.bias_dropout_ratio,
                name='%s/%s/bias_dropout' % (tower_name, bias_tower_cfg.input))

          tf.summary.scalar(
              '%s/%s/bias_output' % (tower_name, bias_tower_cfg.input),
              tf.reduce_mean(bias_logits))
          output_logits = output_logits + bias_logits

      tower_outputs[tower_name] = output_logits

    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict

  def get_shared_ws(self):
    shared_ws = [
        x for x in tf.global_variables()
        if 'bottom_dnn' in x.name and '/kernel' in x.name
    ]
    shared_ws.sort(key=lambda x: x.name)
    return shared_ws[-1]
