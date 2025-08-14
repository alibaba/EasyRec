# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for early stopping."""

import collections
import datetime
import logging
import operator
import os
import threading
import time
from distutils.version import LooseVersion

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

from easy_rec.python.utils.config_util import parse_time
from easy_rec.python.utils.load_class import load_by_path

if LooseVersion(tf.__version__) >= LooseVersion('2.12.0'):
  from tensorflow_estimator.python.estimator.estimator_export import estimator_export
else:
  from tensorflow.python.util.tf_export import estimator_export

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'

EARLY_STOP_SIG_SCOPE = 'signal_early_stopping'
EARLY_STOP_SIG_NAME = 'STOP'


def find_early_stop_var(var_list):
  pattern = EARLY_STOP_SIG_SCOPE + '/' + EARLY_STOP_SIG_NAME
  for var in var_list:
    if pattern in var.name:
      return var
  return None


@estimator_export('estimator.experimental.make_early_stopping_hook')
def make_early_stopping_hook(estimator,
                             should_stop_fn,
                             run_every_secs=60,
                             run_every_steps=None):
  """Creates early-stopping hook.

  Returns a `SessionRunHook` that stops training when `should_stop_fn` returns `True`. Usage example:
  ```python
  estimator = ...
  hook = early_stopping.make_early_stopping_hook(
      estimator, should_stop_fn=make_stop_fn(...))
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    should_stop_fn: `callable`, function that takes no arguments and returns a
      `bool`. If the function returns `True`, stopping will be initiated by the
      chief.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    A `SessionRunHook` that periodically executes `should_stop_fn` and initiates
    early stopping if the function returns `True`.

  Raises:
    TypeError: If `estimator` is not of type `tf.estimator.Estimator`.
    ValueError: If both `run_every_secs` and `run_every_steps` are set.
  """
  if run_every_secs is not None and run_every_steps is not None:
    raise ValueError('Only one of `run_every_secs` and `run_every_steps` must '
                     'be set.')

  if estimator.config.is_chief:
    return _StopOnPredicateHook(should_stop_fn, run_every_secs, run_every_steps)
  else:
    return _CheckForStoppingHook()


@estimator_export('estimator.experimental.stop_if_higher_hook')
def stop_if_higher_hook(estimator,
                        metric_name,
                        threshold,
                        eval_dir=None,
                        min_steps=0,
                        run_every_secs=60,
                        run_every_steps=None):
  """Creates hook to stop if the given metric is higher than the threshold.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if accuracy becomes higher than 0.9.
  hook = early_stopping.stop_if_higher_hook(estimator, "accuracy", 0.9)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. "loss", "accuracy", etc.
    threshold: Numeric threshold for the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric is higher than specified threshold and initiates
    early stopping if true.
  """
  return _stop_if_threshold_crossed_hook(
      estimator=estimator,
      metric_name=metric_name,
      threshold=threshold,
      higher_is_better=True,
      eval_dir=eval_dir,
      min_steps=min_steps,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


@estimator_export('estimator.experimental.stop_if_lower_hook')
def stop_if_lower_hook(estimator,
                       metric_name,
                       threshold,
                       eval_dir=None,
                       min_steps=0,
                       run_every_secs=60,
                       run_every_steps=None):
  """Creates hook to stop if the given metric is lower than the threshold.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if loss becomes lower than 100.
  hook = early_stopping.stop_if_lower_hook(estimator, "loss", 100)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. "loss", "accuracy", etc.
    threshold: Numeric threshold for the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric is lower than specified threshold and initiates
    early stopping if true.
  """
  return _stop_if_threshold_crossed_hook(
      estimator=estimator,
      metric_name=metric_name,
      threshold=threshold,
      higher_is_better=False,
      eval_dir=eval_dir,
      min_steps=min_steps,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


@estimator_export('estimator.experimental.stop_if_no_increase_hook')
def stop_if_no_increase_hook(estimator,
                             metric_name,
                             max_steps_without_increase,
                             eval_dir=None,
                             min_steps=0,
                             run_every_secs=60,
                             run_every_steps=None):
  """Creates hook to stop if metric does not increase within given max steps.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if accuracy does not increase in over 100000 steps.
  hook = early_stopping.stop_if_no_increase_hook(estimator, "accuracy", 100000)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. "loss", "accuracy", etc.
    max_steps_without_increase: `int`, maximum number of training steps with no
      increase in the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric shows no increase over given maximum number of
    training steps, and initiates early stopping if true.
  """
  return _stop_if_no_metric_improvement_hook(
      estimator=estimator,
      metric_name=metric_name,
      max_steps_without_improvement=max_steps_without_increase,
      higher_is_better=True,
      eval_dir=eval_dir,
      min_steps=min_steps,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


def custom_early_stop_hook(estimator,
                           eval_dir,
                           custom_stop_func,
                           custom_stop_func_params,
                           run_every_secs=60,
                           run_every_steps=None):
  """Custom early stop hook for complex early stop conditions.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    custom_stop_func: custom stop function, accept two parameters: eval_results,
      custom_stop_func_params
    custom_stop_func_params: string, parameters to custom_stop_func
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that stop training and
    evaluation under user defined conditions.
  """
  if eval_dir is None:
    eval_dir = estimator.eval_dir()

  if isinstance(custom_stop_func, str) or isinstance(custom_stop_func,
                                                     type(u'')):
    custom_stop_func = load_by_path(custom_stop_func)

  def _custom_stop_fn():
    eval_results = read_eval_metrics(eval_dir)
    return custom_stop_func(eval_results, custom_stop_func_params)

  return make_early_stopping_hook(
      estimator=estimator,
      should_stop_fn=_custom_stop_fn,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


@estimator_export('estimator.experimental.stop_if_no_decrease_hook')
def stop_if_no_decrease_hook(estimator,
                             metric_name,
                             max_steps_without_decrease,
                             eval_dir=None,
                             min_steps=0,
                             run_every_secs=60,
                             run_every_steps=None):
  """Creates hook to stop if metric does not decrease within given max steps.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if loss does not decrease in over 100000 steps.
  hook = early_stopping.stop_if_no_decrease_hook(estimator, "loss", 100000)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. "loss", "accuracy", etc.
    max_steps_without_decrease: `int`, maximum number of training steps with no
      decrease in the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric shows no decrease over given maximum number of
    training steps, and initiates early stopping if true.
  """
  return _stop_if_no_metric_improvement_hook(
      estimator=estimator,
      metric_name=metric_name,
      max_steps_without_improvement=max_steps_without_decrease,
      higher_is_better=False,
      eval_dir=eval_dir,
      min_steps=min_steps,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


def read_eval_metrics(eval_dir):
  """Helper to read eval metrics from eval summary files.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Returns:
    A `dict` with global steps mapping to `dict` of metric names and values.
  """
  eval_metrics_dict = collections.defaultdict(dict)
  for event in _summaries(eval_dir):
    if not event.HasField('summary'):
      continue
    metrics = {}
    for value in event.summary.value:
      if value.HasField('simple_value'):
        metrics[value.tag] = value.simple_value
    if metrics:
      eval_metrics_dict[event.step].update(metrics)
  return collections.OrderedDict(
      sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


def _stop_if_threshold_crossed_hook(estimator, metric_name, threshold,
                                    higher_is_better, eval_dir, min_steps,
                                    run_every_secs, run_every_steps):
  """Creates early-stopping hook to stop training if threshold is crossed."""
  if eval_dir is None:
    eval_dir = estimator.eval_dir()

  is_lhs_better = operator.gt if higher_is_better else operator.lt
  greater_or_lesser = 'greater than' if higher_is_better else 'less than'

  def stop_if_threshold_crossed_fn():
    """Returns `True` if the given metric crosses specified threshold."""
    eval_results = read_eval_metrics(eval_dir)

    for step, metrics in eval_results.items():
      if step < min_steps:
        continue
      val = metrics[metric_name]
      if is_lhs_better(val, threshold):
        tf_logging.info(
            'At step %s, metric "%s" has value %s which is %s the configured '
            'threshold (%s) for early stopping.', step, metric_name, val,
            greater_or_lesser, threshold)
        return True
    return False

  return make_early_stopping_hook(
      estimator=estimator,
      should_stop_fn=stop_if_threshold_crossed_fn,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


def _stop_if_no_metric_improvement_hook(estimator, metric_name,
                                        max_steps_without_improvement,
                                        higher_is_better, eval_dir, min_steps,
                                        run_every_secs, run_every_steps):
  """Returns hook to stop training if given metric shows no improvement."""
  if eval_dir is None:
    eval_dir = estimator.eval_dir()

  is_lhs_better = operator.gt if higher_is_better else operator.lt
  increase_or_decrease = 'increase' if higher_is_better else 'decrease'

  def stop_if_no_metric_improvement_fn():
    """Returns `True` if metric does not improve within max steps."""
    eval_results = read_eval_metrics(eval_dir)

    best_val = None
    best_val_step = None
    for step, metrics in eval_results.items():
      if step < min_steps:
        continue
      val = metrics[metric_name]
      if best_val is None or is_lhs_better(val, best_val):
        best_val = val
        best_val_step = step
      if step - best_val_step >= max_steps_without_improvement:
        tf_logging.info(
            'No %s in metric "%s" for %s steps, which is greater than or equal '
            'to max steps (%s) configured for early stopping.',
            increase_or_decrease, metric_name, step - best_val_step,
            max_steps_without_improvement)
        return True
    return False

  return make_early_stopping_hook(
      estimator=estimator,
      should_stop_fn=stop_if_no_metric_improvement_fn,
      run_every_secs=run_every_secs,
      run_every_steps=run_every_steps)


def _summaries(eval_dir):
  """Yields `tensorflow.Event` protos from event files in the eval dir.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Yields:
    `tensorflow.Event` object read from the event files.
  """
  if gfile.Exists(eval_dir):
    for event_file in gfile.Glob(
        os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
      for event in summary_iterator.summary_iterator(event_file):
        yield event


def _get_or_create_stop_var():
  with variable_scope.variable_scope(
      name_or_scope=EARLY_STOP_SIG_SCOPE,
      values=[],
      reuse=variable_scope.AUTO_REUSE):
    return variable_scope.get_variable(
        name=EARLY_STOP_SIG_NAME,
        shape=[],
        dtype=dtypes.bool,
        initializer=init_ops.constant_initializer(False),
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        trainable=False)


class _StopOnPredicateHook(session_run_hook.SessionRunHook):
  """Hook that requests stop when `should_stop_fn` returns `True`."""

  def __init__(self, should_stop_fn, run_every_secs=60, run_every_steps=None):
    if not callable(should_stop_fn):
      raise TypeError('`should_stop_fn` must be callable.')

    self._should_stop_fn = should_stop_fn
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_secs=run_every_secs, every_steps=run_every_steps)
    self._global_step_tensor = None
    self._stop_var = _get_or_create_stop_var()
    self._stop_op = None

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    self._stop_op = state_ops.assign(self._stop_var, True)

  def before_run(self, run_context):
    del run_context
    return session_run_hook.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      if self._should_stop_fn():
        tf_logging.info('Requesting early stopping at global step %d',
                        global_step)
        run_context.session.run(self._stop_op)
        run_context.request_stop()


class _CheckForStoppingHook(session_run_hook.SessionRunHook):
  """Hook that requests stop if stop is requested by `_StopOnPredicateHook`."""

  def __init__(self):
    self._stop_var = None

  def begin(self):
    self._stop_var = _get_or_create_stop_var()

  def before_run(self, run_context):
    del run_context
    return session_run_hook.SessionRunArgs(self._stop_var)

  def after_run(self, run_context, run_values):
    should_early_stop = run_values.results
    if should_early_stop:
      tf_logging.info('Early stopping requested, suspending run.')
      run_context.request_stop()


class OssStopSignalHook(session_run_hook.SessionRunHook):

  def __init__(self, model_dir, run_every_secs=10, run_every_steps=None):
    self._stop_sig_file = os.path.join(model_dir, 'OSS_STOP_SIGNAL')
    self._stop = False
    self._check_run = True
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_secs=run_every_secs, every_steps=run_every_steps)
    sleep_time = run_every_secs if run_every_secs is not None else 1
    self._curr_step = 0

    def _check_stop():
      while self._check_run:
        if self._timer.should_trigger_for_step(self._curr_step):
          self._timer.update_last_triggered_step(self._curr_step)
          if gfile.Exists(self._stop_sig_file):
            self._stop = True
            logging.info('OssStopSignalHook: stop on signal %s' %
                         self._stop_sig_file)
            break
        else:
          time.sleep(sleep_time)

    self._th = threading.Thread(target=_check_stop)
    self._th.start()

    self._global_step_tensor = None
    self._stop_var = _get_or_create_stop_var()
    self._stop_op = None

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    self._stop_op = state_ops.assign(self._stop_var, True)

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    if self._stop:
      run_context.request_stop()
      run_context.session.run(self._stop_op)
    self._curr_step = run_values.results

  def end(self, session):
    self._check_run = False
    self._th.join()


def oss_stop_hook(estimator, run_every_secs=10, run_every_steps=None):
  """Creates oss stop hook.

  Returns a `SessionRunHook` that stops training when model_dir/OSS_STOP_SIGNAL is created.
  """
  if estimator.config.is_chief:
    return OssStopSignalHook(
        estimator.model_dir,
        run_every_secs=run_every_secs,
        run_every_steps=run_every_steps)
  else:
    return _CheckForStoppingHook()


class DeadlineStopHook(session_run_hook.SessionRunHook):

  def __init__(self, deadline_ts):
    self._deadline_ts = deadline_ts
    self._stop_var = _get_or_create_stop_var()
    self._stop_op = None

  def begin(self):
    self._stop_op = state_ops.assign(self._stop_var, True)

  def after_run(self, run_context, run_values):
    curr_ts = time.mktime(datetime.datetime.now().timetuple())
    if curr_ts > self._deadline_ts:
      run_context.request_stop()
      run_context.session.run(self._stop_op)


def deadline_stop_hook(estimator, dead_line):
  """Creates oss stop hook.

  Returns a `SessionRunHook` that stops training when timestamp > deadline_ts.
  """
  deadline_ts = parse_time(dead_line)
  if estimator.config.is_chief:
    return DeadlineStopHook(deadline_ts)
  else:
    return _CheckForStoppingHook()
