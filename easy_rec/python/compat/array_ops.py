import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_math_ops


def convert_to_int_tensor(tensor, name, dtype=tf.int32):
  """Converts the given value to an integer Tensor."""
  tensor = ops.convert_to_tensor(tensor, name=name, preferred_dtype=dtype)
  if tensor.dtype.is_integer:
    tensor = gen_math_ops.cast(tensor, dtype)
  else:
    raise TypeError('%s must be an integer tensor; dtype=%s' %
                    (name, tensor.dtype))
  return tensor


def _with_nonzero_rank(data):
  """If `data` is scalar, then add a dimension; otherwise return as-is."""
  if data.shape.ndims is not None:
    if data.shape.ndims == 0:
      return tf.stack([data])
    else:
      return data
  else:
    data_shape = tf.shape(data)
    data_ndims = tf.rank(data)
    return tf.reshape(data, tf.concat([[1], data_shape], axis=0)[-data_ndims:])


def get_positive_axis(axis, ndims):
  """Validate an `axis` parameter, and normalize it to be positive.

  If `ndims` is known (i.e., not `None`), then check that `axis` is in the
  range `-ndims <= axis < ndims`, and return `axis` (if `axis >= 0`) or
  `axis + ndims` (otherwise).
  If `ndims` is not known, and `axis` is positive, then return it as-is.
  If `ndims` is not known, and `axis` is negative, then report an error.

  Args:
    axis: An integer constant
    ndims: An integer constant, or `None`

  Returns:
    The normalized `axis` value.

  Raises:
    ValueError: If `axis` is out-of-bounds, or if `axis` is negative and
      `ndims is None`.
  """
  if not isinstance(axis, int):
    raise TypeError('axis must be an int; got %s' % type(axis).__name__)
  if ndims is not None:
    if 0 <= axis < ndims:
      return axis
    elif -ndims <= axis < 0:
      return axis + ndims
    else:
      raise ValueError('axis=%s out of bounds: expected %s<=axis<%s' %
                       (axis, -ndims, ndims))
  elif axis < 0:
    raise ValueError('axis may only be negative if ndims is statically known.')
  return axis


def tile_one_dimension(data, axis, multiple):
  """Tiles a single dimension of a tensor."""
  # Assumes axis is a nonnegative int.
  if data.shape.ndims is not None:
    multiples = [1] * data.shape.ndims
    multiples[axis] = multiple
  else:
    ones_value = tf.ones(tf.rank(data), tf.int32)
    multiples = tf.concat(
        [ones_value[:axis], [multiple], ones_value[axis + 1:]], axis=0)
  return tf.tile(data, multiples)


def _all_dimensions(x):
  """Returns a 1D-tensor listing all dimensions in x."""
  # Fast path: avoid creating Rank and Range ops if ndims is known.
  if isinstance(x, ops.Tensor) and x.get_shape().ndims is not None:
    return constant_op.constant(np.arange(x.get_shape().ndims), dtype=tf.int32)
  if (isinstance(x, sparse_tensor.SparseTensor) and
      x.dense_shape.get_shape().is_fully_defined()):
    r = x.dense_shape.get_shape().dims[0].value  # sparse.dense_shape is 1-D.
    return constant_op.constant(np.arange(r), dtype=tf.int32)

  # Otherwise, we rely on `range` and `rank` to do the right thing at runtime.
  return gen_math_ops._range(0, tf.rank(x), 1)


# This op is intended to exactly match the semantics of numpy.repeat, with
# one exception: numpy.repeat has special (and somewhat non-intuitive) behavior
# when axis is not specified.  Rather than implement that special behavior, we
# simply make `axis` be a required argument.
#
# External (OSS) `tf.repeat` feature request:
# https://github.com/tensorflow/tensorflow/issues/8246
def repeat_with_axis(data, repeats, axis, name=None):
  """Repeats elements of `data`.

  Args:
    data: An `N`-dimensional tensor.
    repeats: A 1-D integer tensor specifying how many times each element in
      `axis` should be repeated.  `len(repeats)` must equal `data.shape[axis]`.
      Supports broadcasting from a scalar value.
    axis: `int`.  The axis along which to repeat values.  Must be less than
      `max(N, 1)`.
    name: A name for the operation.

  Returns:
    A tensor with `max(N, 1)` dimensions.  Has the same shape as `data`,
    except that dimension `axis` has size `sum(repeats)`.
  #### Examples:
    ```python
    >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
    ['a', 'a', 'a', 'c', 'c']
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
    [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
    [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
    ```
  """
  if not isinstance(axis, int):
    raise TypeError('axis must be an int; got %s' % type(axis).__name__)

  with ops.name_scope(name, 'Repeat', [data, repeats]):
    data = ops.convert_to_tensor(data, name='data')
    repeats = convert_to_int_tensor(repeats, name='repeats')
    repeats.shape.with_rank_at_most(1)

    # If `data` is a scalar, then upgrade it to a vector.
    data = _with_nonzero_rank(data)
    data_shape = tf.shape(data)

    # If `axis` is negative, then convert it to a positive value.
    axis = get_positive_axis(axis, data.shape.ndims)

    # Check data Tensor shapes.
    if repeats.shape.ndims == 1:
      data.shape.dims[axis].assert_is_compatible_with(repeats.shape[0])

    # If we know that `repeats` is a scalar, then we can just tile & reshape.
    if repeats.shape.ndims == 0:
      expanded = tf.expand_dims(data, axis + 1)
      tiled = tile_one_dimension(expanded, axis + 1, repeats)
      result_shape = tf.concat([data_shape[:axis], [-1], data_shape[axis + 1:]],
                               axis=0)
      return tf.reshape(tiled, result_shape)

    # Broadcast the `repeats` tensor so rank(repeats) == axis + 1.
    if repeats.shape.ndims != axis + 1:
      repeats_shape = tf.shape(repeats)
      repeats_ndims = tf.rank(repeats)
      broadcast_shape = tf.concat(
          [data_shape[:axis + 1 - repeats_ndims], repeats_shape], axis=0)
      repeats = tf.broadcast_to(repeats, broadcast_shape)
      repeats.set_shape([None] * (axis + 1))

    # Create a "sequence mask" based on `repeats`, where slices across `axis`
    # contain one `True` value for each repetition.  E.g., if
    # `repeats = [3, 1, 2]`, then `mask = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]`.
    max_repeat = gen_math_ops.maximum(
        0, gen_math_ops._max(repeats, _all_dimensions(repeats)))
    mask = tf.sequence_mask(repeats, max_repeat)

    # Add a new dimension around each value that needs to be repeated, and
    # then tile that new dimension to match the maximum number of repetitions.
    expanded = tf.expand_dims(data, axis + 1)
    tiled = tile_one_dimension(expanded, axis + 1, max_repeat)

    # Use `boolean_mask` to discard the extra repeated values.  This also
    # flattens all dimensions up through `axis`.
    masked = tf.boolean_mask(tiled, mask)

    # Reshape the output tensor to add the outer dimensions back.
    if axis == 0:
      result = masked
    else:
      result_shape = tf.concat([data_shape[:axis], [-1], data_shape[axis + 1:]],
                               axis=0)
      result = tf.reshape(masked, result_shape)

    # Preserve shape information.
    if data.shape.ndims is not None:
      new_axis_size = 0 if repeats.shape[0] == 0 else None
      result.set_shape(data.shape[:axis].concatenate(
          [new_axis_size]).concatenate(data.shape[axis + 1:]))

    return result


def repeat(input, repeats, axis=None, name=None):  # pylint: disable=redefined-builtin
  """Repeat elements of `input`.

  Args:
    input: An `N`-dimensional Tensor.
    repeats: An 1-D `int` Tensor. The number of repetitions for each element.
      repeats is broadcasted to fit the shape of the given axis. `len(repeats)`
      must equal `input.shape[axis]` if axis is not None.
    axis: An int. The axis along which to repeat values. By default (axis=None),
      use the flattened input array, and return a flat output array.
    name: A name for the operation.

  Returns:
    A Tensor which has the same shape as `input`, except along the given axis.
      If axis is None then the output array is flattened to match the flattened
      input array.
  #### Examples:
    ```python
    >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
    ['a', 'a', 'a', 'c', 'c']
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
    [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
    [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
    >>> repeat(3, repeats=4)
    [3, 3, 3, 3]
    >>> repeat([[1,2], [3,4]], repeats=2)
    [1, 1, 2, 2, 3, 3, 4, 4]
    ```
  """
  if axis is None:
    input = tf.reshape(input, [-1])
    axis = 0
  return repeat_with_axis(input, repeats, axis, name)
