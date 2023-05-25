# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from easy_rec.python.layers.common_layers import SENet
from easy_rec.python.layers.common_layers import BiLinear
from easy_rec.python.layers import dnn


if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class FiBiNetLayer(object):
  """FiBiNet++:Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction.

  This is almost an exact implementation of the original FiBiNet++ model.
  See the original paper:
  https://arxiv.org/pdf/2209.05016.pdf
  """

  def __init__(self, fibinet_config, features, input_layer):
    self._config = fibinet_config
    self._input_layer = input_layer
    self._features = features

  def __call__(self, group_name, is_training, l2_reg=0, *args, **kwargs):
    feature_list = []
    _, group_features = self._input_layer(self._features, group_name)
    senet = SENet(reduction_ratio=self._config.senet_reduction_ratio,
                       num_groups=self._config.num_senet_squeeze_group,
                       name='%s_senet' % group_name)
    senet_output = senet(group_features)
    feature_list.append(senet_output)

    if self._config.bilinear_type != 'none':
      bilinear = BiLinear(output_size=self._config.bilinear_output_units,
                          bilinear_type=self._config.bilinear_type,
                          bilinear_plus=self._config.use_bilinear_plus,
                          name='%s_bilinear' % group_name)
      bilinear_output = bilinear(group_features)
      feature_list.append(bilinear_output)

    if len(feature_list) > 1:
      feature = tf.concat(feature_list, axis=-1)
    else:
      feature = feature_list[0]

    final_dnn = dnn.DNN(
      self._config.mlp,
      l2_reg,
      name='%s_fibinet_mlp' % group_name,
      is_training=is_training)
    return final_dnn(feature)
