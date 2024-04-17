# -*- encoding:utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils used to manipulate tensor shapes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from easy_rec.python.utils import static_shape


def _is_tensor(t):
  """Returns a boolean indicating whether the input is a tensor.

  Args:
    t: the input to be tested.

  Returns:
    a boolean that indicates whether t is a tensor.
  """
  return isinstance(t, (tf.Tensor, tf.SparseTensor, tf.Variable))


def _set_dim_0(t, d0):
  """Sets the 0-th dimension of the input tensor.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    d0: an integer indicating the 0-th dimension of the input tensor.

  Returns:
    the tensor t with the 0-th dimension set.
  """
  t_shape = t.get_shape().as_list()
  t_shape[0] = d0
  t.set_shape(t_shape)
  return t


def merge_shape(t, shape_list):
  """Merge static shape info into tensor.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    shape_list: a list of shape, the same length of t.get_shape()

  Return:
    the tensor t with shape updated
  """
  t_shape = t.get_shape().as_list()
  assert len(shape_list) == len(
      t_shape), 'input shape size should be the same of the tensor'
  for idx, size in enumerate(shape_list):
    if size is not None:
      t_shape[idx] = size
  t.set_shape(t_shape)
  return t


def pad_tensor(t, length):
  """Pads the input tensor with 0s along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after padding, assuming length <= t.shape[0].

  Returns:
    padded_t: the padded tensor, whose first dimension is length. If the length
      is an integer, the first dimension of padded_t is set to length
      statically.
  """
  t_rank = tf.rank(t)
  t_shape = tf.shape(t)
  t_d0 = t_shape[0]
  pad_d0 = tf.expand_dims(length - t_d0, 0)
  pad_shape = tf.cond(
      tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
      lambda: tf.expand_dims(length - t_d0, 0))
  padded_t = tf.concat([t, tf.zeros(pad_shape, dtype=t.dtype)], 0)
  if not _is_tensor(length):
    padded_t = _set_dim_0(padded_t, length)
  return padded_t


def clip_tensor(t, length):
  """Clips the input tensor along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after clipping, assuming length <= t.shape[0].

  Returns:
    clipped_t: the clipped tensor, whose first dimension is length. If the
      length is an integer, the first dimension of clipped_t is set to length
      statically.
  """
  clipped_t = tf.gather(t, tf.range(length))
  if not _is_tensor(length):
    clipped_t = _set_dim_0(clipped_t, length)
  return clipped_t


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after processing.

  Returns:
    processed_t: the processed tensor, whose first dimension is length. If the
      length is an integer, the first dimension of the processed tensor is set
      to length statically.
  """
  return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_nd(tensor, output_shape):
  """Pad given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)

  trailing_paddings = [
      shape - tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [tf.zeros(len(trailing_paddings), dtype=tf.int32), trailing_paddings],
      axis=1)
  padded_tensor = tf.pad(tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor, begin=tf.zeros(len(clip_size), dtype=tf.int32), size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  return pad_nd(clipped_tensor, output_shape)


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def check_min_image_dim(min_dim, image_tensor):
  """Checks that the image width/height are greater than some number.

  This function is used to check that the width and height of an image are above
  a certain value. If the image shape is static, this function will perform the
  check at graph construction time. Otherwise, if the image shape varies, an
  Assertion control dependency will be added to the graph.

  Args:
    min_dim: The minimum number of pixels along the width and height of the
             image.
    image_tensor: The image tensor to check size for.

  Returns:
    If `image_tensor` has dynamic size, return `image_tensor` with a Assert
    control dependency. Otherwise returns image_tensor.

  Raises:
    ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
  """
  image_shape = image_tensor.get_shape()
  image_height = static_shape.get_height(image_shape)
  image_width = static_shape.get_width(image_shape)
  if image_height is None or image_width is None:
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
            tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
        ['image size must be >= {} in both height and width.'.format(min_dim)])
    with tf.control_dependencies([shape_assert]):
      return tf.identity(image_tensor)

  if image_height < min_dim or image_width < min_dim:
    raise ValueError(
        'image size must be >= %d in both height and width; image dim = %d,%d' %
        (min_dim, image_height, image_width))

  return image_tensor


def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else:
      return tf.no_op()
  else:
    return tf.assert_equal(shape_a, shape_b)


def assert_shape_equal_along_first_dimension(shape_a, shape_b):
  """Asserts that shape_a and shape_b are the same along the 0th-dimension.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
    if shape_a[0] != shape_b[0]:
      raise ValueError('Unequal first dimension {}, {}'.format(
          shape_a[0], shape_b[0]))
    else:
      return tf.no_op()
  else:
    return tf.assert_equal(shape_a[0], shape_b[0])


def assert_box_normalized(boxes, maximum_normalized_coordinate=1.1):
  """Asserts the input box tensor is normalized.

  Args:
    boxes: a tensor of shape [N, 4] where N is the number of boxes.
    maximum_normalized_coordinate: Maximum coordinate value to be considered
      as normalized, default to 1.1.

  Returns:
    a tf.Assert op which fails when the input box tensor is not normalized.

  Raises:
    ValueError: When the input box tensor is not normalized.
  """
  box_minimum = tf.reduce_min(boxes)
  box_maximum = tf.reduce_max(boxes)
  return tf.Assert(
      tf.logical_and(
          tf.less_equal(box_maximum, maximum_normalized_coordinate),
          tf.greater_equal(box_minimum, 0)), [boxes])


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None and not tf.executing_eagerly():
    name = tensor.name
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)
  shape = tensor.shape.as_list()
  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)
  if not non_static_indexes:
    return shape
  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None and not tf.executing_eagerly():
    name = tensor.name
  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True
  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        'For the tensor `%s` in scope `%s`, the actual rank '
        '`%d` (shape = %s) is not equal to the expected rank `%s`' %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def truncate_sequence(seq_emb, seq_len, limited_len):

  def truncate(seq_embed, seq_length):
    seq_embed = tf.slice(seq_embed, [0, 0, 0],
                         [shape[0], limited_len, shape[2]])
    seq_length = tf.where(
        tf.greater(seq_length, limited_len),
        tf.ones_like(seq_length) * limited_len, seq_length)
    return seq_embed, seq_length

  def keep(seq_embed, seq_length):
    return seq_embed, seq_length

  shape = get_shape_list(seq_emb)
  max_seq_len = shape[1]

  return tf.cond(max_seq_len > limited_len, lambda: truncate(seq_emb, seq_len),
                 lambda: keep(seq_emb, seq_len))


def pad_or_truncate_sequence(seq_emb, seq_len, fixed_len):
  padding_length = fixed_len - tf.shape(seq_emb)[1]

  def padding():
    paddings = tf.stack([[0, 0], [0, padding_length], [0, 0]])
    padded = tf.pad(seq_emb, paddings)
    return padded, seq_len

  def truncate():
    sliced = tf.slice(seq_emb, [0, 0, 0], [-1, fixed_len, -1])
    length = tf.where(seq_len < fixed_len, seq_len,
                      tf.ones_like(seq_len) *
                      fixed_len) if seq_len is not None else None
    return sliced, length

  def keep():
    return seq_emb, seq_len

  return tf.cond(padding_length > 0, padding,
                 lambda: tf.cond(padding_length < 0, truncate, keep))
