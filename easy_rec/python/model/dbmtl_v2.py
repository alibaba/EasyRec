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


class DBMTLV2(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(DBMTLV2, self).__init__(model_config, feature_configs, features, labels,
                                is_training)
    assert self._model_config.WhichOneof('model') == 'dbmtl', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dbmtl
    assert isinstance(self._model_config, DBMTLConfig)

    self._init_towers(self._model_config.task_towers)

  def build_predict_graph(self):
    task_input_list = []
    for i, task_tower_cfg in enumerate(self._model_config.task_towers):
      tower_name = task_tower_cfg.tower_name
      with tf.variable_scope('bottom/%s' % tower_name) as scope:
        scope.shared_var_collection_name = tower_name
        bottom_fea, _ = self._input_layer(self._feature_dict, 'all')
        if self._model_config.HasField('bottom_dnn'):
          bottom_dnn = dnn.DNN(
              self._model_config.bottom_dnn,
              self._l2_reg,
              name='bottom_dnn',
              is_training=self._is_training)
          bottom_fea = bottom_dnn(bottom_fea)
      tf.summary.scalar('bottom_fea/%s' % tower_name, tf.norm(bottom_fea))
      task_input_list.append(bottom_fea)

    if self._model_config.use_sequence_encoder:
      for i, task_tower_cfg in enumerate(self._model_config.task_towers):
        tower_name = task_tower_cfg.tower_name
        with tf.variable_scope('sequence/%s' % tower_name):
          seq_encoding = self.get_sequence_encoding(
              is_training=self._is_training)
          tf.summary.scalar('seq_norm/%s' % tower_name, tf.norm(seq_encoding) / tf.to_float(tf.shape(seq_encoding)[0]))
          task_input_list[i] = tf.concat([bottom_fea, seq_encoding], axis=-1)

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
