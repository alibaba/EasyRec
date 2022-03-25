# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

from easy_rec.python.vision_backbones.nets import resnet_utils
from easy_rec.python.vision_backbones import net_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim

def image_mask(valid_shape,
               max_shape=None,
               dtype=tf.bool,
               name=None):

  """Returns a image mask tensor.
  Args:
    valid_shape: integer tensor of shape `(batch_size, 2 or 3)`, image valid shape
    max_shape: integer tensor of shape `(2 or 3)`, size of y x dimension of returned tensor.
    dtype: output type of the resulting tensor.
    name: name of the op.
  Returns:
    A mask tensor of shape `(batch_size, max_shape[0], max_shape[1])`, cast to specified dtype.
  """
  with tf.name_scope(name or "ImageMask"):
    if max_shape is None:
      max_shape = tf.reduce_max(valid_shape, axis=0)

    x = tf.range(max_shape[1])
    y = tf.range(max_shape[0])
    X, Y = tf.meshgrid(x, y)

    valid_shape_x = valid_shape[:, 1]
    X = X[tf.newaxis, :, :] < valid_shape_x[:, tf.newaxis, tf.newaxis]

    valid_shape_y = valid_shape[:, 0]
    Y = Y[tf.newaxis, :, :] < valid_shape_y[:, tf.newaxis, tf.newaxis]

    return tf.cast(tf.logical_and(X, Y), dtype=dtype)


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

@slim.add_arg_scope
def squeeze_and_excitation_2d(inputs,
                              se_rate=16,
                              inputs_mask=None):
  """
  squeeze and excitation block
  ref to Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-Excitation Networks. CoRR.

  Args:
    inputs: input tensor of size [batch_size, height, width, channels].
    se_rate: squeeze-and-excitation reduce rate.
    inputs_mask: input tensor valid mask of size [batch_size, height, width].

  Returns:
    output tensor with same shape as inputs
  """

  input_shape = combined_static_and_dynamic_shape(inputs)
  input_c = input_shape[-1]

  if inputs_mask is not None:
    assert inputs_mask.shape.ndims == inputs.shape.ndims
    assert inputs_mask.dtype == inputs.dtype
    with tf.variable_scope('global_pool'):
      valid_sum = tf.reduce_sum(inputs_mask, axis=[1, 2], keep_dims=True)
      input = inputs * inputs_mask
      conv_dw = tf.reduce_sum(input,
                              axis=[1, 2],
                              keep_dims=True) / valid_sum
  else:
    conv_dw = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name='global_pool')
  conv_dw = slim.conv2d(inputs=conv_dw,
                        num_outputs=input_c // se_rate,
                        kernel_size=(1, 1),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope="conv_down")
  conv_up = slim.conv2d(inputs=conv_dw,
                        num_outputs=input_c,
                        kernel_size=(1, 1),
                        activation_fn=tf.sigmoid,
                        normalizer_fn=None,
                        scope='conv_up')
  conv = tf.multiply(inputs, conv_up, name='excite')
  return conv

class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


@slim.add_arg_scope
def basic(inputs,
          depth,
          stride,
          rate=1,
          avg_down=False,
          se_rate=None,
          inputs_mask=None,
          outputs_collections=None,
          scope=None,
          use_bounded_activations=False):
  """Basic residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    avg_down: bool, default False
      Whether to use average pooling for projection skip connection between stages/downsample.
    se_rate: reduce rate for squeeze_and_excitation_2d, if None, not use SE.
    inputs_mask: inputs valid mask. A tensor of size [batch, height, width].
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'basic_v1', [inputs]) as sc:
    shortcut = resnet_utils.shortcut(inputs, depth=depth, stride=stride, avg_down=avg_down,
                                     use_bounded_activations=use_bounded_activations,
                                     scope='shortcut')
    residual = resnet_utils.conv2d_same(inputs, depth, 3, stride=stride, rate=rate, scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth, 3, stride=1, rate=rate,
                                        activation_fn=None, scope='conv2')

    if se_rate is not None:
      residual = squeeze_and_excitation_2d(inputs=residual,
                                                         se_rate=se_rate,
                                                         inputs_mask=inputs_mask)

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               middle_stride=True,
               avg_down=False,
               se_rate=None,
               inputs_mask=None,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    middle_stride: bool, default True, Whether stride is set on 3x3 layer or not.
    avg_down: bool, default False
      Whether to use average pooling for projection skip connection between stages/downsample.
    se_rate: reduce rate for squeeze_and_excitation_2d, if None, not use SE.
    inputs_mask: inputs valid mask. A tensor of size [batch, height, width].
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1 if middle_stride else stride,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride=stride if middle_stride else 1,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')
    shortcut_depth = slim.utils.last_dimension(residual.get_shape(), min_rank=4)
    shortcut = resnet_utils.shortcut(inputs, depth=shortcut_depth, stride=stride, avg_down=avg_down,
                                     scope='shortcut')


    if se_rate is not None:
      residual = squeeze_and_excitation_2d(inputs=residual,
                                                         se_rate=se_rate,
                                                         inputs_mask=inputs_mask)

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              deep_stem=False,
              inputs_true_shape=None,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode. If this is set
      to None, the callers can specify slim.batch_norm's is_training parameter
      from an outer slim.arg_scope.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction,
      None for skip average pooling.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
      activations at the last unit of each block and store them in the
      `outputs_collections` before subsampling them. This gives us access to
      higher resolution intermediate activations which are useful in some
      dense prediction problems but increases 4x the computation and memory cost
      at the last unit of each block.
    deep_stem: bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    inputs_true_shape: true shape for each example in batch. [batch, 3]
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes a non-zero integer, net contains the
      pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck, basic,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with (slim.arg_scope([slim.batch_norm], is_training=is_training)
      if is_training is not None else NoOpScope()):
        net = inputs
        net_mask = None
        if inputs_true_shape is not None:
          net_mask = image_mask(
              inputs_true_shape, dtype=tf.float32)[:, :, :, tf.newaxis]
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          if deep_stem:
            net = resnet_utils.conv2d_same(net, 32, 3, stride=2, scope='conv1')
            net = resnet_utils.conv2d_same(net, 32, 3, stride=1, scope='conv2')
            net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='conv3')
          else:
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
          if net_mask is not None:
            net_mask = net_mask[:, ::4, ::4, :]

        net = resnet_utils.stack_blocks_dense(net, blocks,
                                              output_stride,
                                              store_non_strided_activations,
                                              net_mask=net_mask)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)

        if global_pool is not None:
          # when global pool is not, skip average pooling
          if global_pool:
            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            end_points['global_pool'] = net
          else:
            # Pooling with a fixed kernel size.
            kernel_size = net_utils.reduced_kernel_size_for_small_input(net, [7, 7])
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a_{}x{}'.format(*kernel_size))
            end_points['AvgPool_1a'] = net

        if num_classes:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


resnet_v1.default_image_size = 224


def resnet_v1_legacy_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 legacy slim-style bottleneck block,
  implemented as a stride in the last unit.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def resnet_v1_basic_block(scope,
                          base_depth,
                          num_units,
                          stride,
                          avg_down=False,
                          se_rate=None):
  """Helper function for creating a resnet_v1 basic block.
  implemented as a stride in the first unit.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the first unit.
      All other units have stride=1.
    avg_down: bool, default False
      Whether to use average pooling for projection skip connection between stages/downsample.
    se_rate: reduce rate for squeeze_and_excitation_2d, if None, not use SE.
  Returns:
    A resnet_v1 basic block.
  """
  return resnet_utils.Block(scope, basic, [{
      'depth': base_depth,
      'stride': stride,
      'avg_down': avg_down,
      'se_rate': se_rate
  }] + [{
      'depth': base_depth,
      'stride': 1,
      'avg_down': avg_down,
      'se_rate': se_rate
  }] * (num_units - 1))


def resnet_v1_bottleneck_block(scope,
                               base_depth,
                               num_units,
                               stride,
                               middle_stride=True,
                               avg_down=False,
                               se_rate=None):
  """Helper function for creating a resnet_v1 bottleneck block.
  implemented as a stride in the first unit.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the first unit.
      All other units have stride=1.
    middle_stride: bool, default True, Whether stride is set on 3x3 layer or not.
    avg_down: bool, default False
      Whether to use average pooling for projection skip connection between stages/downsample.
    se_rate: reduce rate for squeeze_and_excitation_2d, if None, not use SE.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'middle_stride': middle_stride,
      'avg_down': avg_down,
      'se_rate': se_rate
  }] + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'middle_stride': middle_stride,
      'avg_down': avg_down,
      'se_rate': se_rate
  }] * (num_units - 1))


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 inputs_true_shape=None,
                 reuse=None,
                 include_root_block=True,
                 block_names=None,
                 block_kwargs=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_legacy_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=2)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=6, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=1))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1_101'):
  """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_legacy_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=2)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=23, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=1))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1_152'):
  """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_legacy_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=2)),
      ('block2', dict(base_depth=128, num_units=8, stride=2)),
      ('block3', dict(base_depth=256, num_units=36, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=1))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1a_18(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1a_18'):
  """ResNet-18 A-variant model of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_basic_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=2, stride=1)),
      ('block2', dict(base_depth=128, num_units=2, stride=2)),
      ('block3', dict(base_depth=256, num_units=2, stride=2)),
      ('block4', dict(base_depth=512, num_units=2, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1a_34(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1a_34'):
  """ResNet-34 model A-variant  of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_basic_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=6, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1a_50(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1a_50'):
  """ResNet-50 model A-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, middle_stride=False)),
      ('block2', dict(base_depth=128, num_units=4, stride=2, middle_stride=False)),
      ('block3', dict(base_depth=256, num_units=6, stride=2, middle_stride=False)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, middle_stride=False))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1a_101(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1a_101'):
  """ResNet-101 model A-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, middle_stride=False)),
      ('block2', dict(base_depth=128, num_units=4, stride=2, middle_stride=False)),
      ('block3', dict(base_depth=256, num_units=23, stride=2, middle_stride=False)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, middle_stride=False))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1a_152(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1b_152'):
  """ResNet-152 model A-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, middle_stride=False)),
      ('block2', dict(base_depth=128, num_units=8, stride=2, middle_stride=False)),
      ('block3', dict(base_depth=256, num_units=36, stride=2, middle_stride=False)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, middle_stride=False))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1b_50(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1b_50'):
  """ResNet-50 model B-variant of of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=6, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1b_101(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1b_101'):
  """ResNet-101 model B-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=23, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1b_152(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1b_152'):
  """ResNet-152 model B-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=8, stride=2)),
      ('block3', dict(base_depth=256, num_units=36, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1c_50(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1c_50'):
  """ResNet-50 model C-variant of of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=6, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   deep_stem=True, inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1c_101(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1c_101'):
  """ResNet-101 model C-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=4, stride=2)),
      ('block3', dict(base_depth=256, num_units=23, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   deep_stem=True, inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1c_152(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1c_152'):
  """ResNet-152 model C-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1)),
      ('block2', dict(base_depth=128, num_units=8, stride=2)),
      ('block3', dict(base_depth=256, num_units=36, stride=2)),
      ('block4', dict(base_depth=512, num_units=3, stride=2))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   deep_stem=True, inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1d_50(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  inputs_true_shape=None,
                  reuse=None,
                  include_root_block=True,
                  block_names=None,
                  block_kwargs=None,
                  scope='resnet_v1d_50'):
  """ResNet-50 model D-variant of of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, avg_down=True)),
      ('block2', dict(base_depth=128, num_units=4, stride=2, avg_down=True)),
      ('block3', dict(base_depth=256, num_units=6, stride=2, avg_down=True)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, avg_down=True))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   deep_stem=True, inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1d_101(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1d_101'):
  """ResNet-101 model D-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, avg_down=True)),
      ('block2', dict(base_depth=128, num_units=4, stride=2, avg_down=True)),
      ('block3', dict(base_depth=256, num_units=23, stride=2, avg_down=True)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, avg_down=True))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   deep_stem=True, inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def resnet_v1d_152(inputs,
                   num_classes=None,
                   is_training=True,
                   global_pool=True,
                   output_stride=None,
                   spatial_squeeze=True,
                   inputs_true_shape=None,
                   reuse=None,
                   include_root_block=True,
                   block_names=None,
                   block_kwargs=None,
                   scope='resnet_v1d_152'):
  """ResNet-152 model D-variant of [1]. See resnet_v1() for arg and return description."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, avg_down=True)),
      ('block2', dict(base_depth=128, num_units=8, stride=2, avg_down=True)),
      ('block3', dict(base_depth=256, num_units=36, stride=2, avg_down=True)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, avg_down=True))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   deep_stem=True, inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


# def resnet_v1_200(inputs,
#                   num_classes=None,
#                   is_training=True,
#                   global_pool=True,
#                   output_stride=None,
#                   spatial_squeeze=True,
#                   inputs_true_shape=None,
#                   reuse=None,
#                   scope='resnet_v1_200'):
#   """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
#   blocks = [
#     resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
#     resnet_v1_block('block2', base_depth=128, num_units=24, stride=2),
#     resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
#     resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)
#   ]
#   return resnet_v1(inputs, blocks, num_classes, is_training,
#                    global_pool=global_pool, output_stride=output_stride,
#                    include_root_block=True, spatial_squeeze=spatial_squeeze,
#                    inputs_true_shape=inputs_true_shape,
#                    reuse=reuse, scope=scope)


def se_resnet_v1_50(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    inputs_true_shape=None,
                    reuse=None,
                    include_root_block=True,
                    block_names=None,
                    block_kwargs=None,
                    scope='se_resnet_v1_50'):
  """SE ResNet-50 model of [1]."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, middle_stride=False, se_rate=16)),
      ('block2', dict(base_depth=128, num_units=4, stride=2, middle_stride=False, se_rate=16)),
      ('block3', dict(base_depth=256, num_units=6, stride=2, middle_stride=False, se_rate=16)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, middle_stride=False, se_rate=16))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def se_resnet_v1_101(inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     inputs_true_shape=None,
                     reuse=None,
                     include_root_block=True,
                     block_names=None,
                     block_kwargs=None,
                     scope='se_resnet_v1_101'):
  """SE ResNet-101 model of [1]."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, middle_stride=False, se_rate=16)),
      ('block2', dict(base_depth=128, num_units=4, stride=2, middle_stride=False, se_rate=16)),
      ('block3', dict(base_depth=256, num_units=23, stride=2, middle_stride=False, se_rate=16)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, middle_stride=False, se_rate=16))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def se_resnet_v1_152(inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     inputs_true_shape=None,
                     reuse=None,
                     include_root_block=True,
                     block_names=None,
                     block_kwargs=None,
                     scope='se_resnet_v1_152'):
  """SE ResNet-152 model of [1]."""
  block_func = resnet_v1_bottleneck_block
  default_kwargs = OrderedDict([
      ('block1', dict(base_depth=64, num_units=3, stride=1, middle_stride=False, se_rate=16)),
      ('block2', dict(base_depth=128, num_units=8, stride=2, middle_stride=False, se_rate=16)),
      ('block3', dict(base_depth=256, num_units=36, stride=2, middle_stride=False, se_rate=16)),
      ('block4', dict(base_depth=512, num_units=3, stride=2, middle_stride=False, se_rate=16))])
  block_names = block_names if block_names else default_kwargs.keys()
  block_kwargs = block_kwargs if block_kwargs else [{}] * len(block_names)
  blocks = [block_func(block_name, **dict(default_kwargs[block_name], **block_kwarg))
            for block_name, block_kwarg in zip(block_names, block_kwargs)]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=include_root_block, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=True,
                   inputs_true_shape=inputs_true_shape,
                   reuse=reuse, scope=scope)


def get_scopes_of_levels(scope, with_logits=True):
  """
  Args:
    scope: scope name for resnet_v1
    with_logits: with classification layer or not.
  return a list of variable scope list order by levels.
  """
  scopes_of_levels = [[scope + "/block4"],
                      [scope + "/block3"],
                      [scope + "/block2"],
                      [scope + "/block1"],
                      [scope + "/conv1", scope + '/conv2', scope + '/conv3']]
  if with_logits:
    return [[scope + "/logits"]] + scopes_of_levels
  else:
    return scopes_of_levels
