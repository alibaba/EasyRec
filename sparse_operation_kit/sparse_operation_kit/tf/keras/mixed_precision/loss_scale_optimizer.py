"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def _multiply_gradient(gradient, scale):
  """Multiply a (possibly sparse) gradient by the given scale factor."""
  scale = math_ops.cast(scale, gradient.dtype)
  if isinstance(gradient, ops.IndexedSlices):
    return ops.IndexedSlices(
        gradient.values * scale,
        gradient.indices,
        dense_shape=gradient.dense_shape)
  else:
    return gradient * scale

class LossScaleOptimizer(mixed_precision.LossScaleOptimizer):
    """
    Abbreviated as `sok.tf.keras.mixed_precision.LossScaleOptimizer`
    """
    def __init__(self, *args, **kwargs):
        super(LossScaleOptimizer, self).__init__(*args, **kwargs)

    def __getattribute__(self, name):
        if "initializer" == name:
            # for plugin optimizer only
            return self._optimizer.initializer
        else:
            return super(LossScaleOptimizer, self).__getattribute__(name)

    def get_unscaled_gradients(self, grads):
        """Unscales the gradients by the loss scale.

        This method is only needed if you compute gradients manually, e.g. with
        `tf.GradientTape`. In that case, call this method to unscale the gradients
        after computing them with `tf.GradientTape`. If you use
        `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
        scaling is automatically applied and this method is unneeded.

        If this method is called, `get_scaled_loss` should also be called. See
        the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
        example.

        Args:
        grads: A list of tensors, each which will be divided by the loss scale.
            Can have None values, which are ignored.

        Returns:
        A new list the same size as `grads`, where every non-None value in `grads`
        is divided by `LossScaleOptimizer.loss_scale()`.
        """
        loss_scale = self._loss_scale()
        loss_scale_reciprocal = 1. / loss_scale
        return [
            _multiply_gradient(g, loss_scale_reciprocal) if g is not None else None
            for g in grads
        ]