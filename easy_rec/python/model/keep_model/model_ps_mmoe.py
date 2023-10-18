#!/usr/bin/env python
# encoding: utf-8
"""MMOE model."""
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Multiply
from tensorflow.keras.models import Model

# from utils.basic_utils import *
# from model.base_model_ps import BaseModel
from easy_rec.python.model.keep_model.dnn import DNN
from easy_rec.python.model.keep_model.senet import SENETLayer


class CustomizedModel(Model):

  def __init__(self, model_conf, indim, name='export_model'):
    super(CustomizedModel, self).__init__(name=name)
    self.label_conf_list = [
        conf for conf in model_conf['label'] if conf.get('weight', 1) > 0
    ]
    self.experts = model_conf['experts']
    self.expert_layers = model_conf['expert_layers']
    self.tower_layers = model_conf['tower_layers']
    self.need_senet = model_conf.get('need_senet', False)
    if self.need_senet:
      self.field_size = model_conf['field_size']
      self.se_layer = SENETLayer(self.field_size)
    self.need_ppnet = model_conf.get('need_ppnet', False)
    self.pp_feature_cnt = model_conf.get('pp_feature_cnt', 0)
    embed_size = 11

    # self._set_inputs(tf.TensorSpec([None, indim], tf.float32, name='embed_input'))

    self.expert_list = [
        DNN(self.expert_layers,
            activation='relu',
            use_bn=True,
            output_activation='relu',
            need_ppnet=self.need_ppnet,
            pp_size=self.pp_feature_cnt * embed_size)
        for _ in range(self.experts)
    ]

    self.tower_list = [
        DNN(self.tower_layers,
            activation='relu',
            use_bn=True,
            output_activation='sigmoid',
            need_ppnet=self.need_ppnet,
            pp_size=self.pp_feature_cnt * embed_size)
        for _ in self.label_conf_list
    ]

    self.gate_list = [
        DNN([self.experts],
            activation='relu',
            use_bn=False,
            output_activation='linear') for _ in self.label_conf_list
    ]

  def call(self, embed_input, training=False):
    if self.need_senet:
      embed_input = self.se_layer(embed_input)
    expert_list = []
    for expert_layer in self.expert_list:
      # (N, 1, d1)
      expert = tf.expand_dims(expert_layer(embed_input), axis=1)
      expert_list.append(expert)
    # (N, n, d1)
    concat_expert = tf.concat(expert_list, axis=1)

    output_list = []
    for i, label_conf in enumerate(self.label_conf_list):
      # (N, n, 1)
      gate = tf.expand_dims(
          tf.nn.softmax(self.gate_list[i](embed_input)), axis=2)
      # (N, d1)
      output = tf.reduce_sum(concat_expert * gate, axis=1)
      # (N, 1) with softmax
      output = self.tower_list[i](output)
      output_list.append(output)

    return output_list


# class ExpModel(BaseModel):
#     def __init__(self, conf):
#         """
#         初始化参数
#         :param worker_list: 指定的 worker 节点列表
#         :param embed_share: 0 全独立 1 field内共享 2 全共享
#         """
#         super(ExpModel, self).__init__(conf)
#
#     def get_export_model_classs(self):
#         return CustomizedModel
#
