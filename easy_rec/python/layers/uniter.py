# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import multihead_cross_attention
from easy_rec.python.utils.activation import get_activation
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class Uniter(object):
  """UNITER: UNiversal Image-TExt Representation Learning.

  See the original paper:
  https://arxiv.org/abs/1909.11740
  """

  def __init__(self, model_config, feature_configs, features, uniter_config,
               input_layer):
    self._model_config = uniter_config
    tower_num = 0
    self._img_features = None
    if input_layer.has_group('image'):
      self._img_features, _ = input_layer(features, 'image')
      tower_num += 1
    self._general_features = None
    if input_layer.has_group('general'):
      self._general_features, _ = input_layer(features, 'general')
      tower_num += 1
    self._txt_seq_features = None
    if input_layer.has_group('text'):
      self._txt_seq_features, _, _ = input_layer(
          features, 'text', is_combine=False)
      tower_num += 1
    self._use_token_type = True if tower_num > 1 else False
    self._other_features = None
    if input_layer.has_group('other'):  # e.g. statistical feature
      self._other_features, _ = input_layer(features, 'other')
      tower_num += 1
    assert tower_num > 0, 'there must be one of the feature groups: [image, text, general, other]'

    self._general_feature_num = 0
    self._txt_feature_num, self._img_feature_num = 0, 0
    general_feature_names = set()
    img_feature_names, txt_feature_names = set(), set()
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
        self._txt_feature_num = len(fea_group.feature_names)
        txt_feature_names = set(fea_group.feature_names)
        assert self._txt_feature_num == len(txt_feature_names), (
            'there are duplicate features in `text` feature group')

    if self._txt_feature_num > 1 or self._img_feature_num > 1:
      self._use_token_type = True
    self._token_type_vocab_size = self._txt_feature_num
    if self._img_feature_num > 0:
      self._token_type_vocab_size += 1
    if self._general_feature_num > 0:
      self._token_type_vocab_size += 1

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
      if fea_name in txt_feature_names:
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
        'Uniter requires that all `text` feature dimensions must be consistent.'
    )
    unique_dim_num = len(set(img_fea_emb_dim_list))
    assert unique_dim_num <= 1 and len(
        img_fea_emb_dim_list
    ) == self._img_feature_num, (
        'Uniter requires that all `image` feature dimensions must be consistent.'
    )
    unique_dim_num = len(set(general_emb_dim_list))
    assert unique_dim_num <= 1 and len(
        general_emb_dim_list
    ) == self._general_feature_num, (
        'Uniter requires that all `general` feature dimensions must be consistent.'
    )

    if self._txt_feature_num > 0 and uniter_config.use_position_embeddings:
      assert uniter_config.max_position_embeddings > 0, (
          'model config `max_position_embeddings` must be greater than 0. ')
      assert uniter_config.max_position_embeddings >= max_seq_len, (
          'model config `max_position_embeddings` must be greater than or equal to the maximum of all feature config '
          '`max_seq_len`, which is %d' % max_seq_len)

    self._img_emb_size = img_fea_emb_dim_list[0] if img_fea_emb_dim_list else 0
    self._txt_emb_size = txt_fea_emb_dim_list[0] if txt_fea_emb_dim_list else 0
    self._general_emb_size = general_emb_dim_list[
        0] if general_emb_dim_list else 0
    if self._img_features is not None:
      assert self._img_emb_size > 0, '`image` feature dimensions must be greater than 0, set by `raw_input_dim`'

  def text_embeddings(self, token_type_id):
    all_txt_features = []
    input_masks = []
    hidden_size = self._model_config.hidden_size
    if self._general_features is not None:
      general_features = self._general_features
      if self._general_emb_size != hidden_size:
        # Run a linear projection of `hidden_size`
        general_features = tf.reshape(
            general_features, shape=[-1, self._general_emb_size])
        general_features = tf.layers.dense(
            general_features, hidden_size, name='txt_projection')
      general_features = tf.reshape(
          general_features, shape=[-1, self._general_feature_num, hidden_size])

      batch_size = tf.shape(general_features)[0]
      general_features = multihead_cross_attention.embedding_postprocessor(
          general_features,
          use_token_type=self._use_token_type,
          token_type_ids=tf.ones(
              shape=tf.stack([batch_size, self._general_feature_num]),
              dtype=tf.int32) * token_type_id,
          token_type_vocab_size=self._token_type_vocab_size,
          reuse_token_type=tf.AUTO_REUSE,
          use_position_embeddings=False,
          dropout_prob=self._model_config.hidden_dropout_prob)

      all_txt_features.append(general_features)
      mask = tf.ones(
          shape=tf.stack([batch_size, self._general_feature_num]),
          dtype=tf.int32)
      input_masks.append(mask)

    if self._txt_seq_features is not None:

      def dynamic_mask(x, max_len):
        ones = tf.ones(shape=tf.stack([x]), dtype=tf.int32)
        zeros = tf.zeros(shape=tf.stack([max_len - x]), dtype=tf.int32)
        return tf.concat([ones, zeros], axis=0)

      token_type_id += len(all_txt_features)
      for i, (seq_fea, seq_len) in enumerate(self._txt_seq_features):
        batch_size, max_seq_len, emb_size = get_shape_list(seq_fea, 3)
        if emb_size != hidden_size:
          seq_fea = tf.reshape(seq_fea, shape=[-1, emb_size])
          seq_fea = tf.layers.dense(
              seq_fea, hidden_size, name='txt_seq_projection_%d' % i)
          seq_fea = tf.reshape(seq_fea, shape=[-1, max_seq_len, hidden_size])

        seq_fea = multihead_cross_attention.embedding_postprocessor(
            seq_fea,
            use_token_type=self._use_token_type,
            token_type_ids=tf.ones(
                shape=tf.stack([batch_size, max_seq_len]), dtype=tf.int32) *
            (i + token_type_id),
            token_type_vocab_size=self._token_type_vocab_size,
            reuse_token_type=tf.AUTO_REUSE,
            use_position_embeddings=self._model_config.use_position_embeddings,
            max_position_embeddings=self._model_config.max_position_embeddings,
            position_embedding_name='txt_position_embeddings_%d' % i,
            dropout_prob=self._model_config.hidden_dropout_prob)
        all_txt_features.append(seq_fea)

        input_mask = tf.map_fn(
            fn=lambda t: dynamic_mask(t, max_seq_len),
            elems=tf.to_int32(seq_len))
        input_masks.append(input_mask)

    return all_txt_features, input_masks

  def image_embeddings(self):
    if self._img_features is None:
      return None
    hidden_size = self._model_config.hidden_size
    image_features = self._img_features
    if self._img_emb_size != hidden_size:
      # Run a linear projection of `hidden_size`
      image_features = tf.reshape(
          image_features, shape=[-1, self._img_emb_size])
      image_features = tf.layers.dense(
          image_features, hidden_size, name='img_projection')
    image_features = tf.reshape(
        image_features, shape=[-1, self._img_feature_num, hidden_size])

    batch_size = tf.shape(image_features)[0]
    img_fea = multihead_cross_attention.embedding_postprocessor(
        image_features,
        use_token_type=self._use_token_type,
        token_type_ids=tf.zeros(
            shape=tf.stack([batch_size, self._img_feature_num]),
            dtype=tf.int32),
        token_type_vocab_size=self._token_type_vocab_size,
        reuse_token_type=tf.AUTO_REUSE,
        use_position_embeddings=self._model_config.use_position_embeddings,
        max_position_embeddings=self._model_config.max_position_embeddings,
        position_embedding_name='img_position_embeddings',
        dropout_prob=self._model_config.hidden_dropout_prob)
    return img_fea

  def __call__(self, is_training, *args, **kwargs):
    if not is_training:
      self._model_config.hidden_dropout_prob = 0.0
      self._model_config.attention_probs_dropout_prob = 0.0

    sub_modules = []

    img_fea = self.image_embeddings()
    start_token_id = 1 if self._img_feature_num > 0 else 0
    txt_features, txt_masks = self.text_embeddings(start_token_id)

    if img_fea is not None:
      batch_size = tf.shape(img_fea)[0]
    elif txt_features:
      batch_size = tf.shape(txt_features[0])[0]
    else:
      batch_size = None

    hidden_size = self._model_config.hidden_size
    if batch_size is not None:
      all_features = []
      masks = []
      cls_emb = tf.get_variable(name='cls_emb', shape=[1, 1, hidden_size])
      cls_emb = tf.tile(cls_emb, [batch_size, 1, 1])
      all_features.append(cls_emb)

      mask = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32)
      masks.append(mask)

      if img_fea is not None:
        all_features.append(img_fea)
        mask = tf.ones(
            shape=tf.stack([batch_size, self._img_feature_num]), dtype=tf.int32)
        masks.append(mask)

      if txt_features:
        all_features.extend(txt_features)
        masks.extend(txt_masks)

      all_fea = tf.concat(all_features, axis=1)
      input_mask = tf.concat(masks, axis=1)
      attention_mask = multihead_cross_attention.create_attention_mask_from_input_mask(
          from_tensor=all_fea, to_mask=input_mask)
      hidden_act = get_activation(self._model_config.hidden_act)
      attention_fea = multihead_cross_attention.transformer_encoder(
          all_fea,
          hidden_size=hidden_size,
          num_hidden_layers=self._model_config.num_hidden_layers,
          num_attention_heads=self._model_config.num_attention_heads,
          attention_mask=attention_mask,
          intermediate_size=self._model_config.intermediate_size,
          intermediate_act_fn=hidden_act,
          hidden_dropout_prob=self._model_config.hidden_dropout_prob,
          attention_probs_dropout_prob=self._model_config
          .attention_probs_dropout_prob,
          initializer_range=self._model_config.initializer_range,
          name='uniter')  # shape: [batch_size, seq_length, hidden_size]
      print('attention_fea:', attention_fea.shape)
      mm_fea = attention_fea[:, 0, :]  # [CLS] feature
      sub_modules.append(mm_fea)

    if self._other_features is not None:
      if self._model_config.HasField('other_feature_dnn'):
        l2_reg = kwargs['l2_reg'] if 'l2_reg' in kwargs else 0
        other_dnn_layer = dnn.DNN(self._model_config.other_feature_dnn, l2_reg,
                                  'other_dnn', is_training)
        other_fea = other_dnn_layer(self._other_features)
      else:
        other_fea = self._other_features
      sub_modules.append(other_fea)

    if len(sub_modules) == 1:
      return sub_modules[0]
    output = tf.concat(sub_modules, axis=-1)
    return output
