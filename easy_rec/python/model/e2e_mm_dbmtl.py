# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

from easy_rec.python.layers import dnn
from easy_rec.python.layers import mmoe
from easy_rec.python.layers.common_layers import highway
from easy_rec.python.model.multi_task_model import MultiTaskModel

from easy_rec.python.protos.dbmtl_pb2 import E2E_MM_DBMTL as E2E_MM_DBMTL_Config  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class E2E_MM_DBMTL(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(E2E_MM_DBMTL, self).__init__(model_config, feature_configs, features,
                                       labels, is_training)
    assert self._model_config.WhichOneof('model') == 'e2e_mm_dbmtl', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.e2e_mm_dbmtl
    assert isinstance(self._model_config, E2E_MM_DBMTL_Config)

    self._features, _ = self._input_layer(self._feature_dict, 'all')

    if 'img' in self._input_layer._feature_groups:
      self._img_features, _ = self._input_layer(self._feature_dict, 'img')

      for feature_config in feature_configs:
        if feature_config.feature_type == feature_config.ImgFeature:
          assert feature_config.HasField(
              'img_shape'), 'ImgFeature must set img_shape.'
          self.img_width = feature_config.img_shape.width
          self.img_height = feature_config.img_shape.height
          self.img_channel = feature_config.img_shape.channel
          break

    if 'sample_num' in self._input_layer._feature_groups:
      self._sample_idx_fea, _ = self._input_layer(self._feature_dict,
                                                  'sample_num')

    if 'img_emb' in self._input_layer._feature_groups:
      self._img_emb, _ = self._input_layer(self._feature_dict, 'img_emb')
    self._init_towers(self._model_config.task_towers)

  def tune_img_emb(self, img_emb):
    with tf.device('/CPU:0'):
      if self._model_config.HasField('highway_dnn'):
        emb_size = self._model_config.highway_dnn.emb_size
        logging.info('highway_dnn used in img_emb, and emb_size is %s' %
                     emb_size)
        img_emb = highway(
            img_emb, emb_size, activation=tf.nn.relu, scope='highway_dnn')

      elif self._model_config.HasField('img_dnn'):
        logging.info('img_dnn used in img_emb')
        img_dnn = dnn.DNN(
            self._model_config.img_dnn,
            self._l2_reg,
            name='img_dnn',
            is_training=self._is_training)
        img_emb = img_dnn(img_emb)
      else:
        logging.info('not using img_dnn and highway_dnn in img_emb')
      return img_emb

  def img_net(self, img_feature):
    with tf.device('/CPU:0'):
      img_feature = tf.reshape(
          img_feature, (-1, self.img_width, self.img_height, self.img_channel))
      # if self._model_config.img_model.model_name == 'ResNet':
      #   from easy_vision.python.core.backbones.nets.resnet_v1 import resnet_v1a_18
      #   img_logit = resnet_v1a_18(
      #     img_feature, num_classes=self._model_config.img_model.num_classes, is_training=self._is_training)[0]
      # elif self._model_config.img_model.model_name == 'MobileNet':
      #   from easy_vision.python.core.backbones.nets.mobilenet.mobilenet_v3 import mobilenet
      #   img_logit = mobilenet(
      #         img_feature, num_classes=self._model_config.img_model.num_classes, is_training=self._is_training)[0]
      # else:
      # assert False, "img_model must in [ResNet, MobileNet]"

      img_model = ResNet50(
          include_top=True,
          pooling='max',
          classes=self._model_config.img_model.num_classes,
          weights=None)
      img_logit = img_model(img_feature)

      img_emb = self.tune_img_emb(img_logit)
      self._prediction_dict['img_logits_test'] = img_emb
      if 'sample_num' in self._input_layer._feature_groups:
        # 扩展 img_emb, img_logits
        sample_idx_fea = tf.cast(self._sample_idx_fea, tf.int32)
        img_emb_expanded = tf.gather(img_emb, sample_idx_fea)
        img_emb_expanded = tf.squeeze(img_emb_expanded, axis=1)

        img_logit_expanded = tf.gather(img_logit, sample_idx_fea)
        img_logit_expanded = tf.squeeze(img_logit_expanded, axis=1)
        self._prediction_dict['img_logits'] = img_logit_expanded
        return img_emb_expanded
      else:
        return img_emb

  def build_predict_graph(self):
    with tf.device('/CPU:0'):
      base_features = tf.layers.batch_normalization(
          self._features,
          training=self._is_training,
          trainable=True,
          name='base_feature_emb_bn')

      if self._model_config.HasField('bottom_dnn'):
        bottom_dnn = dnn.DNN(
            self._model_config.bottom_dnn,
            self._l2_reg,
            name='bottom_dnn',
            is_training=self._is_training)
        bottom_fea = bottom_dnn(base_features)
      else:
        bottom_fea = base_features

    # for train
    if 'img' in self._input_layer._feature_groups:
      img_emb = self.img_net(self._img_features)
      bottom_fea = tf.concat([img_emb, bottom_fea], axis=-1)
      self._prediction_dict['img_emb'] = tf.reduce_join(
          tf.as_string(img_emb), axis=-1, separator=',')
    # for predict
    elif 'img_emb' in self._input_layer._feature_groups:
      bottom_fea = tf.concat([self._img_emb, bottom_fea], axis=-1)

    with tf.device('/CPU:0'):
      # MMOE block
      if self._model_config.HasField('expert_dnn'):
        mmoe_layer = mmoe.MMOE(
            self._model_config.expert_dnn,
            l2_reg=self._l2_reg,
            num_task=self._task_num,
            num_expert=self._model_config.num_expert)
        task_input_list = mmoe_layer(bottom_fea)
      else:
        task_input_list = [bottom_fea] * self._task_num

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
        tower_outputs[tower_name] = output_logits

    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict

  def build_loss_graph(self):
    """Build loss graph for multi task model."""
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      loss_weight = task_tower_cfg.weight * self._sample_weight

      if hasattr(task_tower_cfg, 'task_space_indicator_label') and \
          task_tower_cfg.HasField('task_space_indicator_label'):
        in_task_space = tf.to_float(
            self._labels[task_tower_cfg.task_space_indicator_label] > 0)
        loss_weight = loss_weight * (
            task_tower_cfg.in_task_space_weight * in_task_space +
            task_tower_cfg.out_task_space_weight * (1 - in_task_space))

      self._loss_dict.update(
          self._build_loss_impl(
              task_tower_cfg.loss_type,
              label_name=self._label_name_dict[tower_name],
              loss_weight=loss_weight,
              num_class=task_tower_cfg.num_class,
              suffix='_%s' % tower_name))

    if self._model_config.img_model.img_loss_weight:
      label = tf.cast(self._labels['cate_label'], tf.int32)
      img_loss = tf.losses.sparse_softmax_cross_entropy(
          labels=label, logits=self._prediction_dict['img_logits'])
      self._loss_dict[
          'weighted_img_loss'] = img_loss * self._model_config.img_model.img_loss_weight
    return self._loss_dict

  def get_outputs(self):
    outputs = []
    for task_tower_cfg in self._task_towers:
      tower_name = task_tower_cfg.tower_name
      outputs.extend(
          self._get_outputs_impl(
              task_tower_cfg.loss_type,
              task_tower_cfg.num_class,
              suffix='_%s' % tower_name))
    if 'img' in self._input_layer._feature_groups:
      outputs.append('img_emb')
      outputs.append('img_logits_test')
    return outputs
