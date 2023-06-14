# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers.common_layers import BiLinear
from easy_rec.python.layers.common_layers import SENet

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class FiBiNetLayer(object):
  """FiBiNet++:Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction.

  This is almost an exact implementation of the original FiBiNet++ model.
  See the original paper:
  https://arxiv.org/pdf/2209.05016.pdf
  """

  def __init__(self, fibinet_config, name='fibinet'):
    self._config = fibinet_config
    self.name = name

  def __call__(self, inputs, is_training, l2_reg=None, *args, **kwargs):
    feature_list = []

    senet = SENet(self._config.senet, name='%s_senet' % self.name)
    senet_output = senet(inputs)
    feature_list.append(senet_output)

    if self._config.HasField('bilinear'):
      conf = self._config.bilinear
      bilinear = BiLinear(
          output_size=conf.num_output_units,
          bilinear_type=conf.type,
          bilinear_plus=conf.use_plus,
          name='%s_bilinear' % self.name)
      bilinear_output = bilinear(inputs)
      feature_list.append(bilinear_output)

    if len(feature_list) > 1:
      feature = tf.concat(feature_list, axis=-1)
    else:
      feature = feature_list[0]

    if self._config.HasField('mlp'):
      final_dnn = dnn.DNN(
          self._config.mlp,
          l2_reg,
          name='%s_fibinet_mlp' % self.name,
          is_training=is_training)
      feature = final_dnn(feature)
    return feature
