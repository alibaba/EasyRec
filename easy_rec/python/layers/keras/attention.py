# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Attention layers that can be used in sequence DNN/CNN models.

This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
Attention is formed by three tensors: Query, Key and Value.
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class Attention(Layer):
  """Dot-product attention layer, a.k.a. Luong-style attention.

  Inputs are a list with 2 or 3 elements:
  1. A `query` tensor of shape `(batch_size, Tq, dim)`.
  2. A `value` tensor of shape `(batch_size, Tv, dim)`.
  3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
      supplied, `value` will be used as a `key`.

  The calculation follows the steps:
  1. Calculate attention scores using `query` and `key` with shape
      `(batch_size, Tq, Tv)`.
  2. Use scores to calculate a softmax distribution with shape
      `(batch_size, Tq, Tv)`.
  3. Use the softmax distribution to create a linear combination of `value`
      with shape `(batch_size, Tq, dim)`.

  Args:
      use_scale: If `True`, will create a scalar variable to scale the
          attention scores.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
          attention scores. Defaults to `0.0`.
      seed: A Python integer to use as random seed in case of `dropout`.
      score_mode: Function to use to compute attention scores, one of
          `{"dot", "concat"}`. `"dot"` refers to the dot product between the
          query and key vectors. `"concat"` refers to the hyperbolic tangent
          of the concatenation of the `query` and `key` vectors.

  Call Args:
      inputs: List of the following tensors:
          - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
          - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
          - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If
              not given, will use `value` for both `key` and `value`, which is
              the most common case.
      mask: List of the following tensors:
          - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.
              If given, the output will be zero at the positions where
              `mask==False`.
          - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.
              If given, will apply the mask such that values at positions
               where `mask==False` do not contribute to the result.
      return_attention_scores: bool, it `True`, returns the attention scores
          (after masking and softmax) as an additional output argument.
      training: Python boolean indicating whether the layer should behave in
          training mode (adding dropout) or in inference mode (no dropout).
      use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds
          a mask such that position `i` cannot attend to positions `j > i`.
          This prevents the flow of information from the future towards the
          past. Defaults to `False`.

  Output:
      Attention outputs of shape `(batch_size, Tq, dim)`.
      (Optional) Attention scores after masking and softmax with shape
          `(batch_size, Tq, Tv)`.
  """

  def __init__(self, params, name='attention', reuse=None, **kwargs):
    super(Attention, self).__init__(name=name, **kwargs)
    self.use_scale = params.get_or_default('use_scale', False)
    self.scale_by_dim = params.get_or_default('scale_by_dim', False)
    self.score_mode = params.get_or_default('score_mode', 'dot')
    if self.score_mode not in ['dot', 'concat']:
      raise ValueError('Invalid value for argument score_mode. '
                       "Expected one of {'dot', 'concat'}. "
                       'Received: score_mode=%s' % self.score_mode)
    self.dropout = params.get_or_default('dropout', 0.0)
    self.seed = params.get_or_default('seed', None)
    self.scale = None
    self.concat_score_weight = None
    self.return_attention_scores = params.get_or_default(
        'return_attention_scores', False)
    self.use_causal_mask = params.get_or_default('use_causal_mask', False)

  def build(self, input_shape):
    self._validate_inputs(input_shape)
    if self.use_scale:
      self.scale = self.add_weight(
          name='scale',
          shape=(),
          initializer='ones',
          dtype=self.dtype,
          trainable=True,
      )
    if self.score_mode == 'concat':
      self.concat_score_weight = self.add_weight(
          name='concat_score_weight',
          shape=(),
          initializer='ones',
          dtype=self.dtype,
          trainable=True,
      )
    self.built = True

  def _calculate_scores(self, query, key):
    """Calculates attention scores as a query-key dot product.

    Args:
        query: Query tensor of shape `(batch_size, Tq, dim)`.
        key: Key tensor of shape `(batch_size, Tv, dim)`.

    Returns:
        Tensor of shape `(batch_size, Tq, Tv)`.
    """
    if self.score_mode == 'dot':
      scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
      if self.scale is not None:
        scores *= self.scale
      elif self.scale_by_dim:
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scores /= tf.math.sqrt(dk)
    elif self.score_mode == 'concat':
      # Reshape tensors to enable broadcasting.
      # Reshape into [batch_size, Tq, 1, dim].
      q_reshaped = tf.expand_dims(query, axis=-2)
      # Reshape into [batch_size, 1, Tv, dim].
      k_reshaped = tf.expand_dims(key, axis=-3)
      if self.scale is not None:
        scores = self.concat_score_weight * tf.reduce_sum(
            tf.tanh(self.scale * (q_reshaped + k_reshaped)), axis=-1)
      else:
        scores = self.concat_score_weight * tf.reduce_sum(
            tf.tanh(q_reshaped + k_reshaped), axis=-1)
    return scores

  def _apply_scores(self, scores, value, scores_mask=None, training=False):
    """Applies attention scores to the given value tensor.

    To use this method in your attention layer, follow the steps:

    * Use `query` tensor of shape `(batch_size, Tq)` and `key` tensor of
        shape `(batch_size, Tv)` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
        `scores_mask`, calculates
        `attention_distribution = softmax(scores)`, then returns
        `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.

    Args:
        scores: Scores float tensor of shape `(batch_size, Tq, Tv)`.
        value: Value tensor of shape `(batch_size, Tv, dim)`.
        scores_mask: A boolean mask tensor of shape `(batch_size, 1, Tv)`
            or `(batch_size, Tq, Tv)`. If given, scores at positions where
            `scores_mask==False` do not contribute to the result. It must
            contain at least one `True` value in each line along the last
            dimension.
        training: Python boolean indicating whether the layer should behave
            in training mode (adding dropout) or in inference mode
            (no dropout).

    Returns:
        Tensor of shape `(batch_size, Tq, dim)`.
        Attention scores after masking and softmax with shape
            `(batch_size, Tq, Tv)`.
    """
    if scores_mask is not None:
      padding_mask = tf.logical_not(scores_mask)
      # Bias so padding positions do not contribute to attention
      # distribution.  Note 65504. is the max float16 value.
      max_value = 65504.0 if scores.dtype == 'float16' else 1.0e9
      scores -= max_value * tf.cast(padding_mask, dtype=scores.dtype)

    weights = tf.nn.softmax(scores, axis=-1)
    if training and self.dropout > 0:
      weights = tf.nn.dropout(weights, 1.0 - self.dropout, seed=self.seed)
    return tf.matmul(weights, value), weights

  def _calculate_score_mask(self, scores, v_mask, use_causal_mask):
    if use_causal_mask:
      # Creates a lower triangular mask, so position i cannot attend to
      # positions j > i. This prevents the flow of information from the
      # future into the past.
      score_shape = tf.shape(scores)
      # causal_mask_shape = [1, Tq, Tv].
      mask_shape = (1, score_shape[-2], score_shape[-1])
      ones_mask = tf.ones(shape=mask_shape, dtype='int32')
      row_index = tf.cumsum(ones_mask, axis=-2)
      col_index = tf.cumsum(ones_mask, axis=-1)
      causal_mask = tf.greater_equal(row_index, col_index)

      if v_mask is not None:
        # Mask of shape [batch_size, 1, Tv].
        v_mask = tf.expand_dims(v_mask, axis=-2)
        return tf.logical_and(v_mask, causal_mask)
      return causal_mask
    else:
      # If not using causal mask, return the value mask as is,
      # or None if the value mask is not provided.
      return v_mask

  def call(
      self,
      inputs,
      mask=None,
      training=False,
  ):
    self._validate_inputs(inputs=inputs, mask=mask)
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v
    q_mask = mask[0] if mask else None
    v_mask = mask[1] if mask else None
    scores = self._calculate_scores(query=q, key=k)
    scores_mask = self._calculate_score_mask(scores, v_mask,
                                             self.use_causal_mask)
    result, attention_scores = self._apply_scores(
        scores=scores, value=v, scores_mask=scores_mask, training=training)
    if q_mask is not None:
      # Mask of shape [batch_size, Tq, 1].
      q_mask = tf.expand_dims(q_mask, axis=-1)
      result *= tf.cast(q_mask, dtype=result.dtype)
    if self.return_attention_scores:
      return result, attention_scores
    return result

  def compute_mask(self, inputs, mask=None):
    self._validate_inputs(inputs=inputs, mask=mask)
    if mask is None or mask[0] is None:
      return None
    return tf.convert_to_tensor(mask[0])

  def compute_output_shape(self, input_shape):
    """Returns shape of value tensor dim, but for query tensor length."""
    return list(input_shape[0][:-1]), input_shape[1][-1]

  def _validate_inputs(self, inputs, mask=None):
    """Validates arguments of the call method."""
    class_name = self.__class__.__name__
    if not isinstance(inputs, list):
      raise ValueError('{class_name} layer must be called on a list of inputs, '
                       'namely [query, value] or [query, value, key]. '
                       'Received: inputs={inputs}.'.format(
                           class_name=class_name, inputs=inputs))
    if len(inputs) < 2 or len(inputs) > 3:
      raise ValueError('%s layer accepts inputs list of length 2 or 3, '
                       'namely [query, value] or [query, value, key]. '
                       'Received length: %d.' % (class_name, len(inputs)))
    if mask is not None:
      if not isinstance(mask, list):
        raise ValueError(
            '{class_name} layer mask must be a list, '
            'namely [query_mask, value_mask]. Received: mask={mask}.'.format(
                class_name=class_name, mask=mask))
      if len(mask) < 2 or len(mask) > 3:
        raise ValueError(
            '{class_name} layer accepts mask list of length 2 or 3. '
            'Received: inputs={inputs}, mask={mask}.'.format(
                class_name=class_name, inputs=inputs, mask=mask))

  def get_config(self):
    base_config = super(Attention, self).get_config()
    config = {
        'use_scale': self.use_scale,
        'score_mode': self.score_mode,
        'dropout': self.dropout,
    }
    return dict(list(base_config.items()) + list(config.items()))
