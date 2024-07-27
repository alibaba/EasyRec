# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.utils.activation import get_activation


class FM(tf.keras.layers.Layer):
  """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

  References
    - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  Input shape.
    - List of 2D tensor with shape: ``(batch_size,embedding_size)``.
    - Or a 3D tensor with shape: ``(batch_size,field_size,embedding_size)``
  Output shape
    - 2D tensor with shape: ``(batch_size, 1)``.
  """

  def __init__(self, params, name='fm', reuse=None, **kwargs):
    super(FM, self).__init__(name=name, **kwargs)
    self.use_variant = params.get_or_default('use_variant', False)

  def call(self, inputs, **kwargs):
    if type(inputs) == list:
      emb_dims = set(map(lambda x: int(x.shape[-1]), inputs))
      if len(emb_dims) != 1:
        dims = ','.join([str(d) for d in emb_dims])
        raise ValueError('all embedding dim must be equal in FM layer:' + dims)
      with tf.name_scope(self.name):
        fea = tf.stack(inputs, axis=1)
    else:
      assert inputs.shape.ndims == 3, 'input of FM layer must be a 3D tensor or a list of 2D tensors'
      fea = inputs

    with tf.name_scope(self.name):
      square_of_sum = tf.square(tf.reduce_sum(fea, axis=1))
      sum_of_square = tf.reduce_sum(tf.square(fea), axis=1)
      cross_term = tf.subtract(square_of_sum, sum_of_square)
      if self.use_variant:
        cross_term = 0.5 * cross_term
      else:
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=-1, keepdims=True)
    return cross_term


class DotInteraction(tf.keras.layers.Layer):
  """Dot interaction layer of DLRM model..

  See theory in the DLRM paper: https://arxiv.org/pdf/1906.00091.pdf,
  section 2.1.3. Sparse activations and dense activations are combined.
  Dot interaction is applied to a batch of input Tensors [e1,...,e_k] of the
  same dimension and the output is a batch of Tensors with all distinct pairwise
  dot products of the form dot(e_i, e_j) for i <= j if self self_interaction is
  True, otherwise dot(e_i, e_j) i < j.

  Attributes:
    self_interaction: Boolean indicating if features should self-interact.
      If it is True, then the diagonal entries of the interaction metric are
      also taken.
    skip_gather: An optimization flag. If it's set then the upper triangle part
      of the dot interaction matrix dot(e_i, e_j) is set to 0. The resulting
      activations will be of dimension [num_features * num_features] from which
      half will be zeros. Otherwise activations will be only lower triangle part
      of the interaction matrix. The later saves space but is much slower.
    name: String name of the layer.
  """

  def __init__(self, params, name=None, reuse=None, **kwargs):
    super(DotInteraction, self).__init__(name=name, **kwargs)
    self._self_interaction = params.get_or_default('self_interaction', False)
    self._skip_gather = params.get_or_default('skip_gather', False)

  def call(self, inputs, **kwargs):
    """Performs the interaction operation on the tensors in the list.

    The tensors represent as transformed dense features and embedded categorical
    features.
    Pre-condition: The tensors should all have the same shape.

    Args:
      inputs: List of features with shapes [batch_size, feature_dim].

    Returns:
      activations: Tensor representing interacted features. It has a dimension
      `num_features * num_features` if skip_gather is True, otherside
      `num_features * (num_features + 1) / 2` if self_interaction is True and
      `num_features * (num_features - 1) / 2` if self_interaction is False.
    """
    if isinstance(inputs, (list, tuple)):
      # concat_features shape: batch_size, num_features, feature_dim
      try:
        concat_features = tf.stack(inputs, axis=1)
      except (ValueError, tf.errors.InvalidArgumentError) as e:
        raise ValueError('Input tensors` dimensions must be equal, original'
                         'error message: {}'.format(e))
    else:
      assert inputs.shape.ndims == 3, 'input of dot func must be a 3D tensor or a list of 2D tensors'
      concat_features = inputs

    batch_size = tf.shape(concat_features)[0]

    # Interact features, select lower-triangular portion, and re-shape.
    xactions = tf.matmul(concat_features, concat_features, transpose_b=True)
    num_features = xactions.shape[-1]
    ones = tf.ones_like(xactions)
    if self._self_interaction:
      # Selecting lower-triangular portion including the diagonal.
      lower_tri_mask = tf.linalg.band_part(ones, -1, 0)
      upper_tri_mask = ones - lower_tri_mask
      out_dim = num_features * (num_features + 1) // 2
    else:
      # Selecting lower-triangular portion not included the diagonal.
      upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
      lower_tri_mask = ones - upper_tri_mask
      out_dim = num_features * (num_features - 1) // 2

    if self._skip_gather:
      # Setting upper triangle part of the interaction matrix to zeros.
      activations = tf.where(
          condition=tf.cast(upper_tri_mask, tf.bool),
          x=tf.zeros_like(xactions),
          y=xactions)
      out_dim = num_features * num_features
    else:
      activations = tf.boolean_mask(xactions, lower_tri_mask)
    activations = tf.reshape(activations, (batch_size, out_dim))
    return activations


class Cross(tf.keras.layers.Layer):
  """Cross Layer in Deep & Cross Network to learn explicit feature interactions.

  A layer that creates explicit and bounded-degree feature interactions
  efficiently. The `call` method accepts `inputs` as a tuple of size 2
  tensors. The first input `x0` is the base layer that contains the original
  features (usually the embedding layer); the second input `xi` is the output
  of the previous `Cross` layer in the stack, i.e., the i-th `Cross`
  layer. For the first `Cross` layer in the stack, x0 = xi.

  The output is x_{i+1} = x0 .* (W * xi + bias + diag_scale * xi) + xi,
  where .* designates elementwise multiplication, W could be a full-rank
  matrix, or a low-rank matrix U*V to reduce the computational cost, and
  diag_scale increases the diagonal of W to improve training stability (
  especially for the low-rank case).

  References:
      1. [R. Wang et al.](https://arxiv.org/pdf/2008.13535.pdf)
        See Eq. (1) for full-rank and Eq. (2) for low-rank version.
      2. [R. Wang et al.](https://arxiv.org/pdf/1708.05123.pdf)

  Example:

      ```python
      # after embedding layer in a functional model:
      input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
      x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6)
      x1 = Cross()(x0, x0)
      x2 = Cross()(x0, x1)
      logits = tf.keras.layers.Dense(units=10)(x2)
      model = tf.keras.Model(input, logits)
      ```

  Args:
      projection_dim: project dimension to reduce the computational cost.
        Default is `None` such that a full (`input_dim` by `input_dim`) matrix
        W is used. If enabled, a low-rank matrix W = U*V will be used, where U
        is of size `input_dim` by `projection_dim` and V is of size
        `projection_dim` by `input_dim`. `projection_dim` need to be smaller
        than `input_dim`/2 to improve the model efficiency. In practice, we've
        observed that `projection_dim` = d/4 consistently preserved the
        accuracy of a full-rank version.
      diag_scale: a non-negative float used to increase the diagonal of the
        kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an
        identity matrix.
      use_bias: whether to add a bias term for this layer. If set to False,
        no bias term will be used.
      preactivation: Activation applied to output matrix of the layer, before
        multiplication with the input. Can be used to control the scale of the
        layer's outputs and improve stability.
      kernel_initializer: Initializer to use on the kernel matrix.
      bias_initializer: Initializer to use on the bias vector.
      kernel_regularizer: Regularizer to use on the kernel matrix.
      bias_regularizer: Regularizer to use on bias vector.

  Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
  Output shape: A single (batch_size, `input_dim`) dimensional output.
  """

  def __init__(self, params, name='cross', reuse=None, **kwargs):
    super(Cross, self).__init__(name=name, **kwargs)
    self._projection_dim = params.get_or_default('projection_dim', None)
    self._diag_scale = params.get_or_default('diag_scale', 0.0)
    self._use_bias = params.get_or_default('use_bias', True)
    preactivation = params.get_or_default('preactivation', None)
    preact = get_activation(preactivation)
    self._preactivation = tf.keras.activations.get(preact)
    kernel_initializer = params.get_or_default('kernel_initializer',
                                               'truncated_normal')
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    bias_initializer = params.get_or_default('bias_initializer', 'zeros')
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    kernel_regularizer = params.get_or_default('kernel_regularizer', None)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    bias_regularizer = params.get_or_default('bias_regularizer', None)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._input_dim = None
    self._supports_masking = True

    if self._diag_scale < 0:  # pytype: disable=unsupported-operands
      raise ValueError(
          '`diag_scale` should be non-negative. Got `diag_scale` = {}'.format(
              self._diag_scale))

  def build(self, input_shape):
    last_dim = input_shape[0][-1]

    if self._projection_dim is None:
      self._dense = tf.keras.layers.Dense(
          last_dim,
          kernel_initializer=_clone_initializer(self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          use_bias=self._use_bias,
          dtype=self.dtype,
          activation=self._preactivation,
      )
    else:
      self._dense_u = tf.keras.layers.Dense(
          self._projection_dim,
          kernel_initializer=_clone_initializer(self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          use_bias=False,
          dtype=self.dtype,
      )
      self._dense_v = tf.keras.layers.Dense(
          last_dim,
          kernel_initializer=_clone_initializer(self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          use_bias=self._use_bias,
          dtype=self.dtype,
          activation=self._preactivation,
      )
    self.built = True

  def call(self, inputs, **kwargs):
    """Computes the feature cross.

    Args:
      inputs: The input tensor(x0, x)
      - x0: The input tensor
      - x: Optional second input tensor. If provided, the layer will compute
        crosses between x0 and x; if not provided, the layer will compute
        crosses between x0 and itself.

    Returns:
     Tensor of crosses.
    """
    if isinstance(inputs, (list, tuple)):
      x0, x = inputs
    else:
      x0, x = inputs, inputs

    if not self.built:
      self.build(x0.shape)

    if x0.shape[-1] != x.shape[-1]:
      raise ValueError(
          '`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x '
          'dimension {}. This case is not supported yet.'.format(
              x0.shape[-1], x.shape[-1]))

    if self._projection_dim is None:
      prod_output = self._dense(x)
    else:
      prod_output = self._dense_v(self._dense_u(x))

    # prod_output = tf.cast(prod_output, self.compute_dtype)

    if self._diag_scale:
      prod_output = prod_output + self._diag_scale * x

    return x0 * prod_output + x

  def get_config(self):
    config = {
        'projection_dim':
            self._projection_dim,
        'diag_scale':
            self._diag_scale,
        'use_bias':
            self._use_bias,
        'preactivation':
            tf.keras.activations.serialize(self._preactivation),
        'kernel_initializer':
            tf.keras.initializers.serialize(self._kernel_initializer),
        'bias_initializer':
            tf.keras.initializers.serialize(self._bias_initializer),
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self._bias_regularizer),
    }
    base_config = super(Cross, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _clone_initializer(initializer):
  return initializer.__class__.from_config(initializer.get_config())
