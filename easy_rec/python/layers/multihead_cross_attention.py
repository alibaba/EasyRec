# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from easy_rec.python.compat.layers import layer_norm as tf_layer_norm
from easy_rec.python.utils.activation import gelu
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def attention_layer(from_tensor,
                    to_tensor,
                    size_per_head,
                    num_attention_heads=1,
                    attention_mask=None,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    reuse=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention is all you Need".
  If `from_tensor` and `to_tensor` are the same, then this is self-attention.
  Each timestep in `from_tensor` attends to the corresponding sequence in `to_tensor`,
  and returns a fixed-width vector.
  This function first projects `from_tensor` into a "query" tensor and `to_tensor` into "key" and "value" tensors.
  These are (effectively) a list of tensors of length `num_attention_heads`, where each tensor is of shape:
  [batch_size, seq_length, size_per_head].
  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.
  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    size_per_head: int. Size of each attention head.
    num_attention_heads: int. Number of attention heads.
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    reuse: whether to reuse this layer

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        'The rank of `from_tensor` must match the rank of `to_tensor`.')

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          'When passing in rank 2 tensors to attention_layer, the values '
          'for `batch_size`, `from_seq_length`, and `to_seq_length` '
          'must all be specified.')

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name='query',
      kernel_initializer=create_initializer(initializer_range),
      reuse=reuse)

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name='key',
      kernel_initializer=create_initializer(initializer_range),
      reuse=reuse)

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name='value',
      kernel_initializer=create_initializer(initializer_range),
      reuse=reuse)

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


def transformer_encoder(input_tensor,
                        attention_mask=None,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        intermediate_act_fn=gelu,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        initializer_range=0.02,
                        reuse=None,
                        name='transformer'):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.
  See the original paper:
  https://arxiv.org/abs/1706.03762
  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    reuse: whether to reuse this encoder
    name: scope name prefix

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        'The hidden size (%d) is not a multiple of the number of attention '
        'heads (%d)' % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError('The width of the input tensor (%d) != hidden size (%d)' %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope('%s_layer_%d' % (name, layer_idx)):
      layer_input = prev_output

      with tf.variable_scope('attention'):
        with tf.variable_scope('self'):
          # [batch_size * from_seq_length, num_attention_heads * size_per_head]
          attention_output = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              size_per_head=attention_head_size,
              num_attention_heads=num_attention_heads,
              attention_mask=attention_mask,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              reuse=reuse)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope('output', reuse=reuse):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope('intermediate', reuse=reuse):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope('output', reuse=reuse):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output

  final_output = reshape_from_matrix(prev_output, input_shape)
  return final_output


def cross_attention_block(from_tensor,
                          to_tensor,
                          layer_idx,
                          size_per_head,
                          cross_attention_mask=None,
                          self_attention_mask=None,
                          num_attention_heads=1,
                          intermediate_size=512,
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          initializer_range=0.02,
                          name=''):
  """Multi-headed cross attention block.

    Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    layer_idx: int. layer id in the Transformer.
    size_per_head: int. Size of each attention head.
    cross_attention_mask: (optional) int32 Tensor of shape [batch_size, from_seq_length,
      to_seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    self_attention_mask: (optional) int32 Tensor of shape [batch_size, from_seq_length,
      from_seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    name: scope name prefix

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  input_shape = get_shape_list(from_tensor, expected_rank=3)
  batch_size = input_shape[0]
  from_seq_length = input_shape[1]

  input_shape = get_shape_list(to_tensor, expected_rank=3)
  to_seq_length = input_shape[1]

  with tf.variable_scope('%scross_layer_%d' % (name, layer_idx)):
    with tf.variable_scope('attention'):
      with tf.variable_scope('cross'):
        # [batch_size * from_seq_length, num_attention_heads * size_per_head]
        cross_attention_output = attention_layer(
            from_tensor=from_tensor,
            to_tensor=to_tensor,
            size_per_head=size_per_head,
            num_attention_heads=num_attention_heads,
            attention_mask=cross_attention_mask,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
            batch_size=batch_size,
            from_seq_length=from_seq_length,
            to_seq_length=to_seq_length)

      with tf.variable_scope('self'):
        # [batch_size * from_seq_length, num_attention_heads * size_per_head]
        self_attention_output = attention_layer(
            from_tensor=cross_attention_output,
            to_tensor=cross_attention_output,
            size_per_head=size_per_head,
            num_attention_heads=num_attention_heads,
            attention_mask=self_attention_mask,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
            batch_size=batch_size,
            from_seq_length=from_seq_length,
            to_seq_length=from_seq_length)

      with tf.variable_scope('output'):
        attention_output = dropout(self_attention_output, hidden_dropout_prob)
        attention_output = layer_norm(attention_output + cross_attention_output)

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.variable_scope('intermediate'):
      intermediate_output = tf.layers.dense(
          attention_output,
          intermediate_size,
          activation=tf.nn.relu,
          kernel_initializer=create_initializer(initializer_range))

    # Down-project back to `hidden_size` then add the residual.
    with tf.variable_scope('output'):
      layer_output = tf.layers.dense(
          intermediate_output,
          num_attention_heads * size_per_head,
          kernel_initializer=create_initializer(initializer_range))
      layer_output = dropout(layer_output, hidden_dropout_prob)
      # [batch_size * from_seq_length, num_attention_heads * size_per_head]
      layer_output = layer_norm(layer_output + attention_output)

  final_output = reshape_from_matrix(
      layer_output,
      [batch_size, from_seq_length, num_attention_heads * size_per_head])
  return final_output  # [batch_size, from_seq_length, num_attention_heads * size_per_head]


def cross_attention_tower(left_tensor,
                          right_tensor,
                          num_hidden_layers=1,
                          num_attention_heads=12,
                          left_size_per_head=64,
                          right_size_per_head=64,
                          left_intermediate_size=0,
                          right_intermediate_size=0,
                          left_input_mask=None,
                          right_input_mask=None,
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          initializer_range=0.02,
                          name=''):
  """Multi-headed, multi layer cross attention block.

    Args:
    left_tensor: float Tensor of shape [batch_size, left_seq_length,
      from_width].
    right_tensor: float Tensor of shape [batch_size, right_seq_length, to_width].
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    left_size_per_head: int. Size of each attention head of left tower.
    right_size_per_head: int. Size of each attention head of right tower.
    left intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer of left tower. Less or equal to 0 means `num_attention_heads
      * left_size_per_head`
    right intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer of right tower. Less or equal to 0 means `num_attention_heads
      * right_size_per_head`
    left_input_mask: the mask for `left_tensor`
    right_input_mask: the mask for `right_tensor`
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    name: scope name prefix

  Returns:
    tuple of float Tensors of shape ([batch_size, left_seq_length, hidden_size],
      [batch_size, right_seq_length, hidden_size]),
      where hidden_size = num_attention_heads * size_per_head

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if left_intermediate_size <= 0:
    left_intermediate_size = num_attention_heads * left_size_per_head
  if right_intermediate_size <= 0:
    right_intermediate_size = num_attention_heads * right_size_per_head

  left_attention_mask = None
  if left_input_mask is not None:
    left_attention_mask = create_attention_mask_from_input_mask(
        left_tensor, left_attention_mask)

  left_2_right_attention_mask = None
  if right_input_mask is not None:
    left_2_right_attention_mask = create_attention_mask_from_input_mask(
        left_tensor, right_input_mask)

  right_attention_mask = None
  if right_input_mask is not None:
    right_attention_mask = create_attention_mask_from_input_mask(
        right_tensor, right_input_mask)

  right_2_left_attention_mask = None
  if left_input_mask is not None:
    right_2_left_attention_mask = create_attention_mask_from_input_mask(
        right_tensor, left_input_mask)

  prev_left_output = left_tensor
  prev_right_output = right_tensor
  for layer_idx in range(num_hidden_layers):
    left_output = cross_attention_block(
        prev_left_output,
        prev_right_output,
        layer_idx,
        num_attention_heads=num_attention_heads,
        size_per_head=left_size_per_head,
        intermediate_size=left_intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        cross_attention_mask=left_2_right_attention_mask,
        self_attention_mask=left_attention_mask,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        initializer_range=initializer_range,
        name='%sleft_to_right_' % name)
    right_output = cross_attention_block(
        prev_right_output,
        prev_left_output,
        layer_idx,
        num_attention_heads=num_attention_heads,
        size_per_head=right_size_per_head,
        intermediate_size=right_intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        cross_attention_mask=right_2_left_attention_mask,
        self_attention_mask=right_attention_mask,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        initializer_range=initializer_range,
        name='%sright_to_left_' % name)
    prev_left_output = left_output
    prev_right_output = right_output
  return prev_left_output, prev_right_output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf_layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError('Input tensor must have at least rank 2. Shape = %s' %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=tf.stack([batch_size, from_seq_length, 1]), dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name='token_type_embeddings',
                            reuse_token_type=None,
                            use_position_embeddings=True,
                            position_embedding_name='position_embeddings',
                            reuse_position_embedding=None,
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    reuse_token_type: bool. Whether to reuse token type embedding variable.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    reuse_position_embedding: bool. Whether to reuse position embedding variable.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError('`token_type_ids` must be specified if'
                       '`use_token_type` is True.')
    with tf.variable_scope('token_type', reuse=reuse_token_type):
      token_type_table = tf.get_variable(
          name=token_type_embedding_name,
          shape=[token_type_vocab_size, width],
          initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      with tf.variable_scope(
          'position_embedding', reuse=reuse_position_embedding):
        full_position_embeddings = tf.get_variable(
            name=position_embedding_name,
            shape=[max_position_embeddings, width],
            initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor
