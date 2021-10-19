# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

from easy_rec.python.layers import dnn
from easy_rec.python.layers import mmoe
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

    self._features, _ = self._input_layer(self._feature_dict, 'all')

    if 'pic' in self._input_layer._feature_groups:
      self._pic_features, _ = self._input_layer(self._feature_dict, 'pic')
      assert self._model_config.dbmtl.pic_dnn, 'pic_dnn must exist when feature group pic exists.'
      pic_fea_flag = False
      for feature_config in feature_configs:
        if feature_config.feature_type == feature_config.PicFeature:
          self.pic_width = feature_config.width
          self.pic_height = feature_config.height
          self.pic_channel = feature_config.channel
          pic_fea_flag = True
          break
      assert pic_fea_flag, 'Type PicFeature must exist when feature group pic exists.'
    self._model_config = self._model_config.dbmtl
    assert isinstance(self._model_config, DBMTLConfig)
    self._init_towers(self._model_config.task_towers)

  def build_predict_graph(self):
    # all_features = self._features
    if 'pic' in self._input_layer._feature_groups:
      pic_feature = tf.reshape(self._pic_features, (-1, 224, 224, 3))
      img_model = ResNet50(include_top=False,
                         pooling='max',
                         weights='imagenet' if self._is_training else None)
      img_emb = img_model(pic_feature) # (?, 2048)

      pic_dnn = dnn.DNN(
          self._model_config.pic_dnn,
          self._l2_reg,
          name='pic_dnn',
          is_training=self._is_training)
      img_emb = pic_dnn(img_emb)
      all_features = tf.concat([img_emb, self._features], axis=-1)
    else:
      all_features = self._features

    if self._model_config.HasField('bottom_dnn'):
      bottom_dnn = dnn.DNN(
          self._model_config.bottom_dnn,
          self._l2_reg,
          name='bottom_dnn',
          is_training=self._is_training)
      # bottom_fea = bottom_dnn(self._features)
      bottom_fea = bottom_dnn(all_features)
    else:
      # bottom_fea = self._features
      bottom_fea = all_features

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
