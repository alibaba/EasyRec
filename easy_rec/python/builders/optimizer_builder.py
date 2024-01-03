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
"""Functions to build training optimizers."""
import logging

import tensorflow as tf

from easy_rec.python.compat import weight_decay_optimizers
from easy_rec.python.core import learning_schedules

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def build(optimizer_config):
  """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')
  optimizer = None

  summary_vars = []
  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=config.decay,
        momentum=config.momentum_optimizer_value,
        epsilon=config.epsilon)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=config.momentum_optimizer_value)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    optimizer = tf.train.AdamOptimizer(
        learning_rate, beta1=config.beta1, beta2=config.beta2)

  if optimizer_type == 'adamw_optimizer':
    config = optimizer_config.adamw_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    logging.info('adamw_optimizer weight_decay = %.8f' % config.weight_decay)
    optimizer = weight_decay_optimizers.AdamWOptimizer(
        weight_decay=config.weight_decay,
        learning_rate=learning_rate,
        beta1=config.beta1,
        beta2=config.beta2)

  if optimizer_type == 'adam_asyncw_optimizer':
    config = optimizer_config.adam_asyncw_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    logging.info('adam_asyncw_optimizer weight_decay = %.8f' %
                 config.weight_decay)
    optimizer = weight_decay_optimizers.AdamAsyncWOptimizer(
        weight_decay=config.weight_decay,
        learning_rate=learning_rate,
        beta1=config.beta1,
        beta2=config.beta2)

  if optimizer_type == 'lazy_adam_optimizer':
    config = optimizer_config.lazy_adam_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    from easy_rec.python.compat.adam_s import AdamOptimizerS
    optimizer = AdamOptimizerS(
        learning_rate=learning_rate, beta1=config.beta1, beta2=config.beta2)

  if optimizer_type == 'momentumw_optimizer':
    config = optimizer_config.momentumw_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    logging.info('momentumw_optimizer weight_decay = %.8f' %
                 config.weight_decay)
    optimizer = weight_decay_optimizers.MomentumWOptimizer(
        weight_decay=config.weight_decay,
        learning_rate=learning_rate,
        momentum=config.momentum_optimizer_value)

  if optimizer_type == 'adagrad_optimizer':
    config = optimizer_config.adagrad_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=config.initial_accumulator_value)

  if optimizer_type == 'adam_async_optimizer':
    config = optimizer_config.adam_async_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    optimizer = tf.train.AdamAsyncOptimizer(
        learning_rate, beta1=config.beta1, beta2=config.beta2)

  if optimizer_type == 'ftrl_optimizer':
    config = optimizer_config.ftrl_optimizer
    learning_rate = _create_learning_rate(config.learning_rate)
    summary_vars.append(learning_rate)
    optimizer = tf.train.FtrlOptimizer(
        learning_rate=learning_rate,
        learning_rate_power=config.learning_rate_power,
        initial_accumulator_value=config.initial_accumulator_value,
        l1_regularization_strength=config.l1_reg,
        l2_regularization_strength=config.l2_reg,
        l2_shrinkage_regularization_strength=config.l2_shrinkage_reg)

  if optimizer is None:
    raise ValueError('Optimizer %s not supported.' % optimizer_type)

  if optimizer_config.use_moving_average:
    optimizer = tf.contrib.opt.MovingAverageOptimizer(
        optimizer, average_decay=optimizer_config.moving_average_decay)

  return optimizer, summary_vars


def _create_learning_rate(learning_rate_config):
  """Create optimizer learning rate based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  learning_rate = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    learning_rate = tf.constant(
        config.learning_rate, dtype=tf.float32, name='learning_rate')

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    learning_rate = learning_schedules.exponential_decay_with_burnin(
        tf.train.get_or_create_global_step(),
        config.initial_learning_rate,
        config.decay_steps,
        config.decay_factor,
        burnin_learning_rate=config.burnin_learning_rate,
        burnin_steps=config.burnin_steps,
        min_learning_rate=config.min_learning_rate,
        staircase=config.staircase)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    learning_rate = learning_schedules.manual_stepping(
        tf.train.get_or_create_global_step(), learning_rate_step_boundaries,
        learning_rate_sequence, config.warmup)

  if learning_rate_type == 'cosine_decay_learning_rate':
    config = learning_rate_config.cosine_decay_learning_rate
    learning_rate = learning_schedules.cosine_decay_with_warmup(
        tf.train.get_or_create_global_step(), config.learning_rate_base,
        config.total_steps, config.warmup_learning_rate, config.warmup_steps,
        config.hold_base_rate_steps)

  if learning_rate_type == 'poly_decay_learning_rate':
    config = learning_rate_config.poly_decay_learning_rate
    learning_rate = tf.train.polynomial_decay(
        config.learning_rate_base, tf.train.get_or_create_global_step(),
        config.total_steps, config.end_learning_rate, config.power)

  if learning_rate_type == 'transformer_learning_rate':
    config = learning_rate_config.transformer_learning_rate
    learning_rate = learning_schedules.transformer_policy(
        tf.train.get_or_create_global_step(), config.learning_rate_base,
        config.hidden_size, config.warmup_steps, config.step_scaling_rate)

  if learning_rate is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return learning_rate
