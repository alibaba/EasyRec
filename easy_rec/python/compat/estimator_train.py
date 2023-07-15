# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import tensorflow as tf
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator.training import _assert_eval_spec
from tensorflow.python.estimator.training import _ContinuousEvalListener
from tensorflow.python.estimator.training import _TrainingExecutor
from tensorflow.python.util import compat

from easy_rec.python.compat.exporter import FinalExporter
from easy_rec.python.utils import estimator_utils

from tensorflow.python.distribute import estimator_training as distribute_coordinator_training  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
gfile = tf.gfile


class TrainDoneListener(_ContinuousEvalListener):
  """Interface for listeners that take action before or after evaluation."""

  def __init__(self, estimator):
    self._model_dir = estimator.model_dir
    self._train_done_file = os.path.join(self._model_dir,
                                         'ESTIMATOR_TRAIN_DONE')

  @property
  def train_done_file(self):
    return self._train_done_file

  def after_eval(self, eval_result):
    """Called after the evaluation is executed.

    Args:
      eval_result: An `_EvalResult` instance.

    Returns:
      False if you want to early stop continuous evaluation; `True` otherwise.
    """
    last_ckpt_path = eval_result.checkpoint_path
    if last_ckpt_path is not None:
      model_dir = os.path.dirname(last_ckpt_path).rstrip('/') + '/'
      latest_ckpt_path = estimator_utils.latest_checkpoint(model_dir)
      if latest_ckpt_path != last_ckpt_path:
        logging.info(
            'TrainDoneListener: latest_ckpt_path[%s] != last_ckpt_path[%s]' %
            (latest_ckpt_path, last_ckpt_path))
        # there are more checkpoints wait to be evaluated
        return True
    return not gfile.Exists(self._train_done_file)


def train_and_evaluate(estimator, train_spec, eval_spec):
  _assert_eval_spec(eval_spec)  # fail fast if eval_spec is invalid.

  train_done_listener = TrainDoneListener(estimator)

  executor = _TrainingExecutor(
      estimator=estimator,
      train_spec=train_spec,
      eval_spec=eval_spec,
      continuous_eval_listener=train_done_listener)
  config = estimator.config

  # If `distribute_coordinator_mode` is set and running in distributed
  # environment, we run `train_and_evaluate` via distribute coordinator.
  if distribute_coordinator_training.should_run_distribute_coordinator(config):
    logging.info('Running `train_and_evaluate` with Distribute Coordinator.')
    distribute_coordinator_training.train_and_evaluate(estimator, train_spec,
                                                       eval_spec,
                                                       _TrainingExecutor)
    return

  if (config.task_type == run_config_lib.TaskType.EVALUATOR and
      config.task_id > 0):
    raise ValueError(
        'For distributed training, there can only be one `evaluator` task '
        '(with task id 0).  Given task id {}'.format(config.task_id))

  result = executor.run()

  # fix for the bug evaluator fails to export in case num_epoch is reached
  # before num_steps is reached or num_steps is set to infinite
  if estimator_utils.is_evaluator():
    export_dir_base = os.path.join(
        compat.as_str_any(estimator.model_dir), compat.as_str_any('export'))
    for exporter in eval_spec.exporters:
      if isinstance(exporter, FinalExporter):
        export_path = os.path.join(
            compat.as_str_any(export_dir_base),
            compat.as_str_any(exporter.name))
        # avoid duplicate export
        if gfile.IsDirectory(export_path + '/'):
          continue
        exporter.export(
            estimator=estimator,
            export_path=export_path,
            checkpoint_path=estimator_utils.latest_checkpoint(
                estimator.model_dir),
            eval_result=None,
            is_the_final_export=True)

  if estimator_utils.is_chief():
    with gfile.GFile(train_done_listener.train_done_file, 'w') as fout:
      fout.write('Train Done.')

  return result


def estimator_train_done(estimator):
  train_done_file = os.path.join(estimator.model_dir, 'ESTIMATOR_TRAIN_DONE')
  return gfile.Exists(train_done_file)
