# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.model.rank_model import RankModel
from easy_rec.python.protos.cmbf_pb2 import CMBF as CMBFConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CMBF(RankModel):
  """CMBF: Cross-Modal-Based Fusion Recommendation Algorithm.
  This is almost an exact implementation of the original CMBF model.
  See the original paper:
  https://www.mdpi.com/1424-8220/21/16/5275
  """
  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(CMBF, self).__init__(model_config, feature_configs, features,
                                  labels, is_training)
    assert self._model_config.WhichOneof('model') == 'cmbf', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')

    self._img_features, _ = self._input_layer(self._feature_dict, 'image')
    self._txt_features, _ = self._input_layer(self._feature_dict, 'text')
    self._other_features = None

    self._txt_feature_num, self._img_feature_num = 0, 0
    txt_feature_names, img_feature_names = None, None
    for fea_group in self._model_config.feature_groups:
      if fea_group.group_name == 'text':
        self._txt_feature_num = len(fea_group.feature_names)
        txt_feature_names = set(fea_group.feature_names)
        assert self._txt_feature_num == len(txt_feature_names), \
          'there are duplicate features in `text` feature group'
      elif fea_group.group_name == 'image':
        self._img_feature_num = len(fea_group.feature_names)
        img_feature_names = set(fea_group.feature_names)
        assert self._img_feature_num == len(img_feature_names), \
          'there are duplicate features in `image` feature group'
      elif fea_group.group_name == 'other':
        self._other_features, _ = self._input_layer(self._feature_dict, 'other')
    assert txt_feature_names is not None, '`text` feature group is required.'
    assert img_feature_names is not None, '`image` feature group is required.'

    self._model_config = self._model_config.cmbf
    assert isinstance(self._model_config, CMBFConfig)

    txt_fea_emb_dim_list = []
    img_fea_emb_dim_list = []
    for feature_config in feature_configs:
      fea_name = feature_config.input_names[0]
      if feature_config.HasField('feature_name'):
        fea_name = feature_config.feature_name
      if fea_name in txt_feature_names:
        txt_fea_emb_dim_list.append(feature_config.embedding_dim)
      if fea_name in img_feature_names:
        img_fea_emb_dim_list.append(feature_config.raw_input_dim)
    assert len(set(txt_fea_emb_dim_list)) == 1 and len(txt_fea_emb_dim_list) == self._txt_feature_num, \
      'CMBF requires that all `text` feature dimensions must be consistent.'
    assert len(set(img_fea_emb_dim_list)) == 1 and len(img_fea_emb_dim_list) == self._img_feature_num, \
      'CMBF requires that all `image` feature dimensions must be consistent.'

    self._img_emb_size = img_fea_emb_dim_list[0]
    self._txt_emb_size = txt_fea_emb_dim_list[0]
    self._head_num = self._model_config.multi_head_num
    self._txt_head_size = self._model_config.text_head_size
    self._img_head_size = self._model_config.image_head_size
    self._img_region_num = self._model_config.image_region_num
    self._img_self_attention_layer_num = self._model_config.image_self_attention_layer_num
    self._txt_self_attention_layer_num = self._model_config.text_self_attention_layer_num
    self._cross_modal_layer_num = self._model_config.cross_modal_layer_num
    print('txt_feature_num: {0}, img_feature_num: {1}'.format(self._txt_feature_num, self._img_feature_num))
    print('txt_embedding_size: {0}, img_embedding_size: {1}'.format(self._txt_emb_size, self._img_emb_size))
    assert self._img_emb_size > 0, '`image` feature dimensions must be greater than 0, set by `raw_input_dim`'

  def build_predict_graph(self):
    hidden_size = self._img_head_size * self._head_num
    image_features = self._img_features
    img_fea_num = self._img_feature_num
    if img_fea_num > 1:  # in case of video frames
      if self._img_emb_size != hidden_size:
        # Run a linear projection of `hidden_size`
        image_features = tf.reshape(self._img_features, shape=[-1, self._img_emb_size])
        image_features = tf.layers.dense(image_features, hidden_size, name='img_projection')
      image_features = tf.reshape(image_features, shape=[-1, self._img_feature_num, hidden_size])
    elif img_fea_num == 1:
      if self._img_region_num > 1:  # image feature: [region_num, emb_size]
        img_fea_num = self._img_region_num
        img_emb_size = self._img_emb_size // self._img_region_num
        assert img_emb_size * self._img_region_num == self._img_emb_size, \
          'image feature dimension must equal to `image_region_num * embedding_size_per_region`'
        self._img_emb_size = img_emb_size
        if self._img_emb_size != hidden_size:
          # Run a linear projection of `hidden_size`
          image_features = tf.reshape(self._img_features, shape=[-1, self._img_emb_size])
          image_features = tf.layers.dense(image_features, hidden_size, name='img_projection')
        image_features = tf.reshape(image_features, shape=[-1, img_fea_num, hidden_size])
      else:  # convert each element of image feature to a feature vector
        # img_fea_num = self._img_emb_size
        img_mapping_matrix = tf.get_variable('img_map_matrix', [1, self._img_emb_size, hidden_size],
                                             dtype=tf.float32)
        image_features = tf.expand_dims(image_features, -1) * img_mapping_matrix

    img_attention_fea = multihead_cross_attention.transformer_encoder(
      image_features,
      hidden_size=hidden_size,  # head_num * size_per_head
      num_hidden_layers=self._img_self_attention_layer_num,
      num_attention_heads=self._head_num,
      intermediate_size=hidden_size*2,
      hidden_dropout_prob=self._model_config.hidden_dropout_prob,
      attention_probs_dropout_prob=self._model_config.attention_probs_dropout_prob,
      name='image_self_attention'
    )  # [batch_size, image_num, hidden_size]
    print('img_attention_fea:', img_attention_fea.shape)

    text_features = self._txt_features
    hidden_size = self._txt_head_size * self._head_num
    if self._txt_emb_size != hidden_size:
      # Run a linear projection of `hidden_size`
      text_features = tf.reshape(text_features, shape=[-1, self._txt_emb_size])
      text_features = tf.layers.dense(text_features, hidden_size, name='txt_projection')
    text_features = tf.reshape(text_features, shape=[-1, self._txt_feature_num, hidden_size])
    txt_attention_fea = multihead_cross_attention.transformer_encoder(
      text_features,
      hidden_size=hidden_size,
      num_hidden_layers=self._txt_self_attention_layer_num,
      num_attention_heads=self._head_num,
      intermediate_size=hidden_size*2,
      hidden_dropout_prob=self._model_config.hidden_dropout_prob,
      attention_probs_dropout_prob=self._model_config.attention_probs_dropout_prob,
      name='text_self_attention'
    )  # [batch_size, txt_seq_length, hidden_size]
    print('txt_attention_fea:', txt_attention_fea.shape)

    img_embeddings, txt_embeddings = multihead_cross_attention.cross_attention_tower(
      img_attention_fea,
      txt_attention_fea,
      num_hidden_layers=self._cross_modal_layer_num,
      num_attention_heads=self._head_num,
      left_size_per_head=self._model_config.image_cross_head_size,
      left_intermediate_size=2 * self._model_config.image_cross_head_size * self._head_num,
      right_size_per_head=self._model_config.text_cross_head_size,
      right_intermediate_size=2 * self._model_config.text_cross_head_size * self._head_num,
      hidden_dropout_prob=self._model_config.hidden_dropout_prob,
      attention_probs_dropout_prob=self._model_config.attention_probs_dropout_prob
    )
    print('img_embeddings:', img_embeddings.shape)
    print('txt_embeddings:', txt_embeddings.shape)

    img_embeddings = tf.reshape(
        img_embeddings,
        shape=[-1, img_embeddings.shape[1] * img_embeddings.shape[2]])
    txt_embeddings = tf.reshape(
      txt_embeddings,
      shape=[-1, txt_embeddings.shape[1] * txt_embeddings.shape[2]])

    all_fea = [img_embeddings, txt_embeddings]
    if self._other_features:
      all_fea.append(self._other_features)
    hidden = tf.concat(all_fea, axis=-1)
    final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg,
                              'final_dnn', self._is_training)
    all_fea = final_dnn_layer(hidden)

    final = tf.layers.dense(all_fea, self._num_class, name='output')
    self._add_to_prediction_dict(final)
    return self._prediction_dict
