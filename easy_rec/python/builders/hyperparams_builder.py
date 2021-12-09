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
"""Builder function to construct tf-slim arg_scope for convolution, fc ops."""
import tensorflow as tf

from easy_rec.python.compat import regularizers

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def build_regularizer(regularizer):
  """Builds a tensorflow regularizer from config.

  Args:
    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tensorflow regularizer.

  Raises:
    ValueError: On unknown regularizer.
  """
  regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
  if regularizer_oneof == 'l1_regularizer':
    return regularizers.l1_regularizer(
        scale=float(regularizer.l1_regularizer.scale))
  if regularizer_oneof == 'l2_regularizer':
    return regularizers.l2_regularizer(
        scale=float(regularizer.l2_regularizer.scale))
  if regularizer_oneof == 'l1_l2_regularizer':
    return regularizers.l1_l2_regularizer(
        scale_l1=float(regularizer.l1_l2_regularizer.scale_l1),
        scale_l2=float(regularizer.l1_l2_regularizer.scale_l2))

  raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))


def build_initializer(initializer):
  """Build a tf initializer from config.

  Args:
    initializer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf initializer.

  Raises:
    ValueError: On unknown initializer.
  """
  initializer_oneof = initializer.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.truncated_normal_initializer(
        mean=initializer.truncated_normal_initializer.mean,
        stddev=initializer.truncated_normal_initializer.stddev)
  if initializer_oneof == 'random_normal_initializer':
    return tf.random_normal_initializer(
        mean=initializer.random_normal_initializer.mean,
        stddev=initializer.random_normal_initializer.stddev)
  if initializer_oneof == 'glorot_normal_initializer':
    return tf.glorot_normal_initializer()
  if initializer_oneof == 'constant_initializer':
    return tf.constant_initializer(
        [x for x in initializer.constant_initializer.consts])
  raise ValueError('Unknown initializer function: {}'.format(initializer_oneof))
