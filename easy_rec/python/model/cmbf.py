# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import cmbf
from easy_rec.python.layers import dnn
from easy_rec.python.model.rank_model import RankModel

from easy_rec.python.protos.cmbf_pb2 import CMBF as CMBFConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CMBF(RankModel):
  """CMBF: Cross-Modal-Based Fusion Recommendation Algorithm.
  This is almost an exact implementation of the original CMBF model.
  See the original paper:
  https://www.mdpi.com/1424-8220/21/16/5275

  Image feature tower can be one of:
  1. multiple image embeddings, each corresponding to video frames or ROIs(region of interest)
  2. one conventional image embedding extracted by a image model
  3. one big image embedding composed by multiple results of spatial convolutions(feature maps before CNN pooling layer)

  If image embedding size is not equal to configured `image_feature_dim` argument,
  do dimension reduce to this size before single modal learning module
  """

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(CMBF, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert self._model_config.WhichOneof('model') == 'cmbf', \
      'invalid model config: %s' % self._model_config.WhichOneof('model')

    self._cmbf_layer = cmbf.CMBF(model_config, feature_configs, features,
                                 self._model_config.cmbf.config,
                                 self._input_layer)
    self._model_config = self._model_config.cmbf

  def build_predict_graph(self):
    hidden = self._cmbf_layer()
    final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                              'final_dnn', self._is_training)
    all_fea = final_dnn_layer(hidden)

    final = tf.layers.dense(all_fea, self._num_class, name='output')
    self._add_to_prediction_dict(final)
    return self._prediction_dict
