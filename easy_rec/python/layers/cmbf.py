# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CMBF(object):
  """CMBF: Cross-Modal-Based Fusion Recommendation Algorithm.

  This is almost an exact implementation of the original CMBF model.
  See the original paper:
  https://www.mdpi.com/1424-8220/21/16/5275
  """

  def __init__(self, model_config, feature_configs, features, cmbf_config,
               input_layer):
    self._model_config = cmbf_config

    has_feature = False
    self._img_features = None
    if input_layer.has_group('image'):
      self._img_features, _ = input_layer(features, 'image')
      has_feature = True
    self._general_features = None
    if input_layer.has_group('general'):
      self._general_features, _ = input_layer(features, 'general')
      has_feature = True
    self._txt_seq_features = None
    if input_layer.has_group('text'):
      self._txt_seq_features, _, _ = input_layer(
          features, 'text', is_combine=False)
      has_feature = True
    self._other_features = None
    if input_layer.has_group('other'):  # e.g. statistical feature
      self._other_features, _ = input_layer(features, 'other')
      has_feature = True
    assert has_feature, 'there must be one of the feature groups: [image, text, general, other]'

    self._general_feature_num, self._img_feature_num = 0, 0
    self._txt_feature_num = 0
    general_feature_names, txt_seq_feature_names = set(), set()
    img_feature_names = set()
    for fea_group in model_config.feature_groups:
      if fea_group.group_name == 'general':
        self._general_feature_num = len(fea_group.feature_names)
        general_feature_names = set(fea_group.feature_names)
        assert self._general_feature_num == len(general_feature_names), (
            'there are duplicate features in `general` feature group')
      elif fea_group.group_name == 'image':
        self._img_feature_num = len(fea_group.feature_names)
        img_feature_names = set(fea_group.feature_names)
        assert self._img_feature_num == len(img_feature_names), (
            'there are duplicate features in `image` feature group')
      elif fea_group.group_name == 'text':
        txt_seq_feature_names = set(fea_group.feature_names)
        self._txt_feature_num = len(fea_group.feature_names)
        assert self._txt_feature_num == len(txt_seq_feature_names), (
            'there are duplicate features in `text` feature group')

    max_seq_len = 0
    txt_fea_emb_dim_list = []
    general_emb_dim_list = []
    img_fea_emb_dim_list = []
    for feature_config in feature_configs:
      fea_name = feature_config.input_names[0]
      if feature_config.HasField('feature_name'):
        fea_name = feature_config.feature_name
      if fea_name in img_feature_names:
        img_fea_emb_dim_list.append(feature_config.raw_input_dim)
      if fea_name in general_feature_names:
        general_emb_dim_list.append(feature_config.embedding_dim)
      if fea_name in txt_seq_feature_names:
        txt_fea_emb_dim_list.append(feature_config.embedding_dim)
        if feature_config.HasField('max_seq_len'):
          assert feature_config.max_seq_len > 0, (
              'feature config `max_seq_len` must be greater than 0 for feature: '
              + fea_name)
          if feature_config.max_seq_len > max_seq_len:
            max_seq_len = feature_config.max_seq_len

    unique_dim_num = len(set(txt_fea_emb_dim_list))
    assert unique_dim_num <= 1 and len(
        txt_fea_emb_dim_list
    ) == self._txt_feature_num, (
        'CMBF requires that all `text` feature dimensions must be consistent.')
    unique_dim_num = len(set(general_emb_dim_list))
    assert unique_dim_num <= 1 and len(
        general_emb_dim_list
    ) == self._general_feature_num, (
        'CMBF requires that all `general` feature dimensions must be consistent.'
    )
    unique_dim_num = len(set(img_fea_emb_dim_list))
    assert unique_dim_num <= 1 and len(
        img_fea_emb_dim_list
    ) == self._img_feature_num, (
        'CMBF requires that all `image` feature dimensions must be consistent.')

    if cmbf_config.use_position_embeddings:
      assert cmbf_config.max_position_embeddings > 0, (
          'model config `max_position_embeddings` must be greater than 0. '
          'It must be set when `use_position_embeddings` is true (default)')
      assert cmbf_config.max_position_embeddings >= max_seq_len, (
          'model config `max_position_embeddings` must be greater than or equal to the maximum of all feature config '
          '`max_seq_len`, which is %d' % max_seq_len)

    self._img_emb_size = img_fea_emb_dim_list[0] if img_fea_emb_dim_list else 0
    self._txt_emb_size = txt_fea_emb_dim_list[0] if txt_fea_emb_dim_list else 0
    self._general_emb_size = general_emb_dim_list[
        0] if general_emb_dim_list else 0
    self._head_num = cmbf_config.multi_head_num
    self._img_head_num = cmbf_config.image_multi_head_num
    self._txt_head_num = cmbf_config.text_multi_head_num
    self._txt_head_size = cmbf_config.text_head_size
    self._img_head_size = cmbf_config.image_head_size
    self._img_patch_num = cmbf_config.image_feature_patch_num
    self._img_self_attention_layer_num = cmbf_config.image_self_attention_layer_num
    self._txt_self_attention_layer_num = cmbf_config.text_self_attention_layer_num
    self._cross_modal_layer_num = cmbf_config.cross_modal_layer_num
    print('txt_feature_num: {0}, img_feature_num: {1}, txt_seq_feature_num: {2}'
          .format(self._general_feature_num, self._img_feature_num,
                  len(self._txt_seq_features) if self._txt_seq_features else 0))
    print('txt_embedding_size: {0}, img_embedding_size: {1}'.format(
        self._txt_emb_size, self._img_emb_size))
    if self._img_features is not None:
      assert self._img_emb_size > 0, '`image` feature dimensions must be greater than 0, set by `raw_input_dim`'

  def image_self_attention_tower(self):
    """The input of image self attention tower can be one of.

    1. multiple image embeddings, each corresponding to a patch, or a ROI(region of interest), or a frame of video
    2. one big image embedding composed by stacking multiple image embeddings
    3. one conventional image embedding extracted by an image model

    If image embedding size is not equal to configured `image_feature_dim` argument,
    do dimension reduce to this size before single modal learning module
    """
    if self._img_features is None:
      return None
    image_features = self._img_features
    img_fea_num = self._img_feature_num
    if self._img_self_attention_layer_num <= 0:
      hidden_size = self._model_config.multi_head_num * self._model_config.image_cross_head_size
      if self._img_emb_size != hidden_size:
        # Run a linear projection of `hidden_size`
        image_features = tf.reshape(
            self._img_features, shape=[-1, self._img_emb_size])
        image_features = tf.layers.dense(
            image_features, hidden_size, name='img_projection')
      image_features = tf.reshape(
          image_features, shape=[-1, img_fea_num, hidden_size])
      return image_features

    hidden_size = self._img_head_size * self._img_head_num
    if img_fea_num > 1:  # in case of video frames or ROIs (Region Of Interest)
      if self._img_emb_size != hidden_size:
        # Run a linear projection of `hidden_size`
        image_features = tf.reshape(
            self._img_features, shape=[-1, self._img_emb_size])
        image_features = tf.layers.dense(
            image_features, hidden_size, name='img_projection')
      image_features = tf.reshape(
          image_features, shape=[-1, img_fea_num, hidden_size])
    elif img_fea_num == 1:
      if self._img_patch_num > 1:  # image feature dimension: patch_num * emb_size
        img_fea_num = self._img_patch_num
        img_emb_size = self._img_emb_size // self._img_patch_num
        assert img_emb_size * self._img_patch_num == self._img_emb_size, (
            'image feature dimension must equal to `image_feature_slice_num * embedding_size_per_region`'
        )
        self._img_emb_size = img_emb_size
        if self._img_emb_size != hidden_size:
          # Run a linear projection of `hidden_size`
          image_features = tf.reshape(
              self._img_features, shape=[-1, self._img_emb_size])
          image_features = tf.layers.dense(
              image_features, hidden_size, name='img_projection')
        image_features = tf.reshape(
            image_features, shape=[-1, img_fea_num, hidden_size])
      else:
        img_fea_num = self._model_config.image_feature_dim
        if img_fea_num != self._img_emb_size:
          image_features = tf.layers.dense(
              image_features, img_fea_num, name='img_projection')
        # convert each element of image feature to a feature vector
        img_mapping_matrix = tf.get_variable(
            'img_map_matrix', [1, img_fea_num, hidden_size], dtype=tf.float32)
        image_features = tf.expand_dims(image_features, -1) * img_mapping_matrix

    img_attention_fea = multihead_cross_attention.transformer_encoder(
        image_features,
        hidden_size=hidden_size,  # head_num * size_per_head
        num_hidden_layers=self._img_self_attention_layer_num,
        num_attention_heads=self._head_num,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=self._model_config.hidden_dropout_prob,
        attention_probs_dropout_prob=self._model_config
        .attention_probs_dropout_prob,
        name='image_self_attention'
    )  # shape: [batch_size, image_seq_num/image_feature_dim, hidden_size]
    # print('img_attention_fea:', img_attention_fea.shape)
    return img_attention_fea

  def text_self_attention_tower(self):
    hidden_size = self._txt_head_size * self._txt_head_num
    txt_features = None
    all_txt_features = []
    input_masks = []

    if self._general_features is not None:
      general_features = self._general_features
      if self._general_emb_size != hidden_size:
        # Run a linear projection of `hidden_size`
        general_features = tf.reshape(
            general_features, shape=[-1, self._general_emb_size])
        general_features = tf.layers.dense(
            general_features, hidden_size, name='txt_projection')
      txt_features = tf.reshape(
          general_features, shape=[-1, self._general_feature_num, hidden_size])

      all_txt_features.append(txt_features)
      batch_size = tf.shape(txt_features)[0]
      mask = tf.ones(
          shape=tf.stack([batch_size, self._general_feature_num]),
          dtype=tf.int32)
      input_masks.append(mask)

    input_mask = None
    attention_mask = None
    if self._txt_seq_features is not None:

      def dynamic_mask(x, max_len):
        ones = tf.ones(shape=tf.stack([x]), dtype=tf.int32)
        zeros = tf.zeros(shape=tf.stack([max_len - x]), dtype=tf.int32)
        return tf.concat([ones, zeros], axis=0)

      token_type_vocab_size = len(self._txt_seq_features)
      for i, (seq_fea, seq_len) in enumerate(self._txt_seq_features):
        batch_size, max_seq_len, emb_size = get_shape_list(seq_fea, 3)
        if emb_size != hidden_size:
          seq_fea = tf.reshape(seq_fea, shape=[-1, emb_size])
          seq_fea = tf.layers.dense(
              seq_fea, hidden_size, name='txt_seq_projection_%d' % i)
          seq_fea = tf.reshape(seq_fea, shape=[-1, max_seq_len, hidden_size])

        seq_fea = multihead_cross_attention.embedding_postprocessor(
            seq_fea,
            use_token_type=self._model_config.use_token_type,
            token_type_ids=tf.ones(
                shape=tf.stack([batch_size, max_seq_len]), dtype=tf.int32) * i,
            token_type_vocab_size=token_type_vocab_size,
            reuse_token_type=tf.AUTO_REUSE,
            use_position_embeddings=self._model_config.use_position_embeddings,
            max_position_embeddings=self._model_config.max_position_embeddings,
            position_embedding_name='position_embeddings_%d' % i,
            dropout_prob=self._model_config.text_seq_emb_dropout_prob)
        all_txt_features.append(seq_fea)

        input_mask = tf.map_fn(
            fn=lambda t: dynamic_mask(t, max_seq_len),
            elems=tf.to_int32(seq_len))
        input_masks.append(input_mask)

      txt_features = tf.concat(all_txt_features, axis=1)
      input_mask = tf.concat(input_masks, axis=1)
      attention_mask = multihead_cross_attention.create_attention_mask_from_input_mask(
          from_tensor=txt_features, to_mask=input_mask)

    if txt_features is None:
      return None, None, None

    txt_attention_fea = multihead_cross_attention.transformer_encoder(
        txt_features,
        hidden_size=hidden_size,
        num_hidden_layers=self._txt_self_attention_layer_num,
        num_attention_heads=self._head_num,
        attention_mask=attention_mask,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=self._model_config.hidden_dropout_prob,
        attention_probs_dropout_prob=self._model_config
        .attention_probs_dropout_prob,
        name='text_self_attention'
    )  # shape: [batch_size, txt_seq_length, hidden_size]
    print('txt_attention_fea:', txt_attention_fea.shape)
    return txt_attention_fea, input_mask, input_masks

  def merge_text_embedding(self, txt_embeddings, input_masks):
    shape = get_shape_list(txt_embeddings)
    if self._txt_seq_features is None:
      return tf.reshape(txt_embeddings, shape=[-1, shape[1] * shape[2]])

    text_seq_emb = []
    if self._general_feature_num > 0:
      text_emb = tf.slice(txt_embeddings, [0, 0, 0],
                          [shape[0], self._general_feature_num, shape[2]])
      text_seq_emb.append(text_emb)

    begin = self._general_feature_num
    for i in range(len(text_seq_emb), len(input_masks)):
      size = tf.shape(input_masks[i])[1]
      temp_emb = tf.slice(txt_embeddings, [0, begin, 0],
                          [shape[0], size, shape[2]])
      mask = tf.expand_dims(tf.to_float(input_masks[i]), -1)
      temp_emb = temp_emb * mask
      # avg pooling
      emb_sum = tf.reduce_sum(
          temp_emb, axis=1,
          keepdims=True)  # shape: [batch_size, 1, hidden_size]
      count = tf.reduce_sum(
          mask, axis=1, keepdims=True)  # shape: [batch_size, 1, 1]
      seq_emb = emb_sum / count  # shape: [batch_size, 1, hidden_size]

      text_seq_emb.append(seq_emb)
      begin = begin + size

    txt_emb = tf.concat(text_seq_emb, axis=1)
    seq_num = len(text_seq_emb)
    if self._general_feature_num > 0:
      seq_num += self._general_feature_num - 1
    txt_embeddings = tf.reshape(txt_emb, shape=[-1, seq_num * shape[2]])
    return txt_embeddings

  def __call__(self, is_training, *args, **kwargs):
    if not is_training:
      self._model_config.hidden_dropout_prob = 0.0
      self._model_config.attention_probs_dropout_prob = 0.0

    # shape: [batch_size, image_num/image_dim, hidden_size]
    img_attention_fea = self.image_self_attention_tower()

    # shape: [batch_size, txt_seq_length, hidden_size]
    txt_attention_fea, input_mask, input_masks = self.text_self_attention_tower(
    )

    all_fea = []
    if None not in [img_attention_fea, txt_attention_fea]:
      img_embeddings, txt_embeddings = multihead_cross_attention.cross_attention_tower(
          img_attention_fea,
          txt_attention_fea,
          num_hidden_layers=self._cross_modal_layer_num,
          num_attention_heads=self._head_num,
          right_input_mask=input_mask,
          left_size_per_head=self._model_config.image_cross_head_size,
          left_intermediate_size=4 * self._model_config.image_cross_head_size *
          self._head_num,
          right_size_per_head=self._model_config.text_cross_head_size,
          right_intermediate_size=4 * self._model_config.text_cross_head_size *
          self._head_num,
          hidden_dropout_prob=self._model_config.hidden_dropout_prob,
          attention_probs_dropout_prob=self._model_config
          .attention_probs_dropout_prob)
      # img_embeddings shape: [batch_size, image_(region_)num/image_feature_dim, multi_head_num * image_cross_head_size]
      print('img_embeddings:', img_embeddings.shape)
      # txt_embeddings shape: [batch_size, general_feature_num + max_txt_seq_len, multi_head_num * text_cross_head_size]
      print('txt_embeddings:', txt_embeddings.shape)

      # shape: [batch_size, multi_head_num * image_cross_head_size]
      img_embeddings = tf.reduce_mean(img_embeddings, axis=1)

      # shape: [batch_size, (general_feature_num + txt_seq_num) * multi_head_num * text_cross_head_size]
      txt_embeddings = self.merge_text_embedding(txt_embeddings, input_masks)
      all_fea = [img_embeddings, txt_embeddings]

    elif img_attention_fea is not None:  # only has image tower
      # avg pooling, shape: [batch_size, multi_head_num * image_head_size]
      img_embeddings = tf.reduce_mean(img_attention_fea, axis=1)
      all_fea = [img_embeddings]

    elif txt_attention_fea is not None:  # only has text tower
      # shape: [batch_size, (general_feature_num + txt_seq_num) * multi_head_num * text_head_size]
      txt_embeddings = self.merge_text_embedding(txt_attention_fea, input_masks)
      all_fea = [txt_embeddings]

    if self._other_features is not None:
      if self._model_config.HasField('other_feature_dnn'):
        l2_reg = kwargs['l2_reg'] if 'l2_reg' in kwargs else 0
        other_dnn_layer = dnn.DNN(self._model_config.other_feature_dnn, l2_reg,
                                  'other_dnn', is_training)
        other_fea = other_dnn_layer(self._other_features)
        all_fea.append(other_fea)  # e.g. statistical features
      else:
        all_fea.append(self._other_features)  # e.g. statistical features

    output = tf.concat(all_fea, axis=-1)
    return output
