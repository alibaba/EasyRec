# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SENet:
  """Squeeze and Excite Network.

  Input shape
  - A list of 2D tensor with shape: ``(batch_size,embedding_size)``.
    The ``embedding_size`` of each field can have different value.

  Args:
    num_fields: int, number of fields.
    num_squeeze_group: int, number of groups for squeeze.
    reduction_ratio: int, reduction ratio for squeeze.
    l2_reg: float, l2 regularizer for embedding.
    name: str, name of the layer.
  """

  def __init__(self,
               num_fields,
               num_squeeze_group,
               reduction_ratio,
               excitation_acitvation,
               l2_reg,
               name='SENet'):
    self.num_fields = num_fields
    self.num_squeeze_group = num_squeeze_group
    self.reduction_ratio = reduction_ratio
    self.excitation_acitvation = excitation_acitvation
    self._l2_reg = l2_reg
    self._name = name

  def __call__(self, inputs):
    r = self.reduction_ratio
    g = self.num_squeeze_group

    emb_size = 0
    for input in inputs:
      emb_size += int(input.shape[-1])

    group_embs = []
    for emb in inputs:
      g_dim = max(2, int(emb.shape[-1]) // g)
      ghat = emb.shape[-1] // g_dim
      group_embs.append(tf.reshape(emb, [-1, ghat, g_dim]))

    squeezed = []
    for emb in group_embs:
      squeezed.append(tf.reduce_mean(emb, axis=-1))  # [B, g]
    z = tf.concat(squeezed, axis=1)  # [bs,  num_groups*field_size]

    reduction_size = max(1, z.shape[-1] // r)

    reduced = tf.layers.dense(
        inputs=z,
        units=reduction_size,
        kernel_regularizer=self._l2_reg,
        activation='relu',
        name='%s/reduce' % self._name)

    excited_weights = tf.layers.dense(
        inputs=reduced,
        units=emb_size,
        kernel_initializer='glorot_normal',
        activation=self.excitation_acitvation,
        name='%s/excite' % self._name)

    # Re-weight
    inputs = tf.concat(inputs, axis=-1)
    output = inputs * excited_weights

    return output
