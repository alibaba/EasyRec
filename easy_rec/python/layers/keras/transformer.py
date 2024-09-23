# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Layer

from easy_rec.python.layers.keras import MultiHeadAttention
from easy_rec.python.layers.keras.layer_norm import LayerNormalization
from easy_rec.python.layers.utils import Parameter
from easy_rec.python.protos import seq_encoder_pb2


class TransformerBlock(Layer):
  """A transformer block combines multi-head attention and feed-forward networks with layer normalization and dropout.

  Purpose: Combines attention and feed-forward layers with residual connections and normalization.
  Components: Multi-head attention, feed-forward network, dropout, and layer normalization.
  Output: Enhanced representation after applying attention and feed-forward layers.
  """

  def __init__(self, params, name='transformer_block', reuse=None, **kwargs):
    super(TransformerBlock, self).__init__(name=name, **kwargs)
    d_model = params.hidden_size
    num_heads = params.num_attention_heads
    mha_cfg = seq_encoder_pb2.MultiHeadAttention()
    mha_cfg.num_heads = num_heads
    mha_cfg.key_dim = d_model // num_heads
    mha_cfg.dropout = params.get_or_default('attention_probs_dropout_prob', 0.0)
    mha_cfg.return_attention_scores = False
    args = Parameter.make_from_pb(mha_cfg)
    self.mha = MultiHeadAttention(args, 'multi_head_attn')
    dropout_rate = params.get_or_default('hidden_dropout_prob', 0.1)
    ffn_units = params.get_or_default('intermediate_size', d_model)
    ffn_act = params.get_or_default('hidden_act', 'relu')
    self.ffn_dense1 = Dense(ffn_units, activation=ffn_act)
    self.ffn_dense2 = Dense(d_model)
    if tf.__version__ >= '2.0':
      self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    else:
      self.layer_norm1 = LayerNormalization(epsilon=1e-6)
      self.layer_norm2 = LayerNormalization(epsilon=1e-6)
    self.dropout1 = Dropout(dropout_rate)
    self.dropout2 = Dropout(dropout_rate)

  def call(self, inputs, training=None, **kwargs):
    x, mask = inputs
    attn_output = self.mha([x, x, x], mask=mask, training=training)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layer_norm1(x + attn_output)
    ffn_mid = self.ffn_dense1(out1)
    ffn_output = self.ffn_dense2(ffn_mid)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layer_norm2(out1 + ffn_output)
    return out2


# Positional Encoding, https://www.tensorflow.org/text/tutorials/transformer
def positional_encoding(length, depth):
  depth = depth / 2
  positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
  angle_rates = 1 / (10000**depths)  # (1, depth)
  angle_rads = positions * angle_rates  # (pos, depth)
  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(Layer):

  def __init__(self, vocab_size, d_model, max_position, name='pos_embedding'):
    super(PositionalEmbedding, self).__init__(name=name)
    self.d_model = d_model
    self.embedding = Embedding(vocab_size, d_model)
    self.pos_encoding = positional_encoding(length=max_position, depth=d_model)

  def call(self, x, training=None):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positional_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class TransformerEncoder(Layer):
  """The encoder consists of a stack of encoder layers.

  It converts the input sequence into a set of embeddings enriched with positional information.
  Purpose: Encodes the input sequence into a set of embeddings.
  Components: Embedding layer, positional encoding, and a stack of transformer blocks.
  Output: Encoded representation of the input sequence.
  """

  def __init__(self, params, name='transformer_encoder', reuse=None, **kwargs):
    super(TransformerEncoder, self).__init__(name=name, **kwargs)
    d_model = params.hidden_size
    dropout_rate = params.get_or_default('hidden_dropout_prob', 0.1)
    max_position = params.get_or_default('max_position_embeddings', 512)
    num_layers = params.get_or_default('num_hidden_layers', 1)
    vocab_size = params.vocab_size
    logging.info('vocab size of TransformerEncoder(%s) is %d', name, vocab_size)
    self.output_all = params.get_or_default('output_all_token_embeddings', True)
    self.pos_encoding = PositionalEmbedding(vocab_size, d_model, max_position)
    self.dropout = Dropout(dropout_rate)
    self.enc_layers = [
        TransformerBlock(params, 'layer_%d' % i) for i in range(num_layers)
    ]
    self._vocab_size = vocab_size
    self._max_position = max_position

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def max_position(self):
    return self._max_position

  def call(self, inputs, training=None, **kwargs):
    x, mask = inputs
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_encoding(x)  # Shape `(batch_size, seq_len, d_model)`.
    x = self.dropout(x, training=training)
    for block in self.enc_layers:
      x = block([x, mask], training)
    # x Shape `(batch_size, seq_len, d_model)`.
    return x if self.output_all else x[:, 0, :]


class TextEncoder(Layer):

  def __init__(self, params, name='text_encoder', reuse=None, **kwargs):
    super(TextEncoder, self).__init__(name=name, **kwargs)
    self.separator = params.get_or_default('separator', ' ')
    self.cls_token = '[CLS]' + self.separator
    self.sep_token = self.separator + '[SEP]' + self.separator
    params.transformer.output_all_token_embeddings = False
    trans_params = Parameter.make_from_pb(params.transformer)
    vocab_file = params.get_or_default('vocab_file', None)
    self.vocab = None
    self.default_token_id = params.get_or_default('default_token_id', 0)
    if vocab_file is not None:
      self.vocab = tf.feature_column.categorical_column_with_vocabulary_file(
          'tokens',
          vocabulary_file=vocab_file,
          default_value=self.default_token_id)
      logging.info('vocab file of TextEncoder(%s) is %s', name, vocab_file)
      trans_params.vocab_size = self.vocab.vocabulary_size
    self.encoder = TransformerEncoder(trans_params, name='transformer')

  def call(self, inputs, training=None, **kwargs):
    if type(inputs) not in (tuple, list):
      inputs = [inputs]
    inputs = [tf.squeeze(text) for text in inputs]
    batch_size = tf.shape(inputs[0])
    cls = tf.fill(batch_size, self.cls_token)
    sep = tf.fill(batch_size, self.sep_token)
    sentences = [cls]
    for sentence in inputs:
      sentences.append(sentence)
      sentences.append(sep)
    text = tf.strings.join(sentences)
    tokens = tf.strings.split(text, self.separator)
    if self.vocab is not None:
      features = {'tokens': tokens}
      token_ids = self.vocab._transform_feature(features)
      token_ids = tf.sparse.to_dense(
          token_ids, default_value=self.default_token_id, name='token_ids')
      length = tf.shape(token_ids)[-1]
      token_ids = tf.cond(
          tf.less_equal(length, self.encoder.max_position), lambda: token_ids,
          lambda: tf.slice(token_ids, [0, 0], [-1, self.encoder.max_position]))
      mask = tf.not_equal(token_ids, self.default_token_id, name='mask')
    else:
      tokens = tf.sparse.to_dense(tokens, default_value='')
      length = tf.shape(tokens)[-1]
      tokens = tf.cond(
          tf.less_equal(length, self.encoder.max_position), lambda: tokens,
          lambda: tf.slice(tokens, [0, 0], [-1, self.encoder.max_position]))
      token_ids = tf.string_to_hash_bucket_fast(
          tokens, self.encoder.vocab_size, name='token_ids')
      mask = tf.not_equal(tokens, '', name='mask')

    encoding = self.encoder([token_ids, mask], training=training)
    return encoding
