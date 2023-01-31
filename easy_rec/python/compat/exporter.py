# -*- encoding:utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""`Exporter` class represents different flavors of model export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from tensorflow.python.estimator import gc
from tensorflow.python.estimator import util
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.exporter import Exporter
from tensorflow.python.estimator.exporter import _SavedModelExporter
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary_iterator

from easy_rec.python.utils import config_util
from easy_rec.python.utils import io_util
from easy_rec.python.utils.export_big_model import export_big_model_to_oss


def _loss_smaller(best_eval_result, current_eval_result):
  """Compares two evaluation results and returns true if the 2nd one is smaller.

  Both evaluation results should have the values for MetricKeys.LOSS, which are
  used for comparison.

  Args:
    best_eval_result: best eval metrics.
    current_eval_result: current eval metrics.

  Returns:
    True if the loss of current_eval_result is smaller; otherwise, False.

  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
  default_key = metric_keys.MetricKeys.LOSS
  if not best_eval_result or default_key not in best_eval_result:
    raise ValueError(
        'best_eval_result cannot be empty or no loss is found in it.')

  if not current_eval_result or default_key not in current_eval_result:
    raise ValueError(
        'current_eval_result cannot be empty or no loss is found in it.')

  return best_eval_result[default_key] > current_eval_result[default_key]


def _verify_compare_fn_args(compare_fn):
  """Verifies compare_fn arguments."""
  args = set(util.fn_args(compare_fn))
  if 'best_eval_result' not in args:
    raise ValueError('compare_fn (%s) must include best_eval_result argument.' %
                     compare_fn)
  if 'current_eval_result' not in args:
    raise ValueError(
        'compare_fn (%s) must include current_eval_result argument.' %
        compare_fn)
  non_valid_args = list(args - set(['best_eval_result', 'current_eval_result']))
  if non_valid_args:
    raise ValueError('compare_fn (%s) has following not expected args: %s' %
                     (compare_fn, non_valid_args))


def _get_ckpt_version(path):
  _, tmp_name = os.path.split(path)
  tmp_name, _ = os.path.splitext(tmp_name)
  ver = tmp_name.split('-')[-1]
  return int(ver)


class BestExporter(Exporter):
  """This class exports the serving graph and checkpoints of the best models.

  This class performs a model export everytime the new model is better than any
  existing model.
  """

  def __init__(self,
               name='best_exporter',
               serving_input_receiver_fn=None,
               event_file_pattern='eval_val/*.tfevents.*',
               compare_fn=_loss_smaller,
               assets_extra=None,
               as_text=False,
               exports_to_keep=5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Example of creating a BestExporter for training and evaluation:

    ```python
    def make_train_and_eval_fn():
      # Set up feature columns.
      categorical_feature_a = (
          tf.feature_column.categorical_column_with_hash_bucket(...))
      categorical_feature_a_emb = embedding_column(
          categorical_column=categorical_feature_a, ...)
      ...  # other feature columns

      estimator = tf.estimator.DNNClassifier(
          config=tf.estimator.RunConfig(
              model_dir='/my_model', save_summary_steps=100),
          feature_columns=[categorical_feature_a_emb, ...],
          hidden_units=[1024, 512, 256])

      serving_feature_spec = tf.feature_column.make_parse_example_spec(
          categorical_feature_a_emb)
      serving_input_receiver_fn = (
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
          serving_feature_spec))

      exporter = tf.estimator.BestExporter(
          name="best_exporter",
          serving_input_receiver_fn=serving_input_receiver_fn,
          exports_to_keep=5)

      train_spec = tf.estimator.TrainSpec(...)

      eval_spec = [tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=100,
        exporters=exporter,
        start_delay_secs=0,
        throttle_secs=5)]

      return tf.estimator.DistributedTrainingSpec(estimator, train_spec,
                                                  eval_spec)
    ```

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      event_file_pattern: event file name pattern relative to model_dir. If
        None, however, the exporter would not be preemption-safe. To be
        preemption-safe, event_file_pattern must be specified.
      compare_fn: a function that compares two evaluation results and returns
        true if current evaluation result is better. Follows the signature:
        * Args:
          * `best_eval_result`: This is the evaluation result of the best model.
          * `current_eval_result`: This is the evaluation result of current
                 candidate model.
        * Returns:
          True if current evaluation result is better; otherwise, False.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as `{'my_asset_file.txt':
        '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any argument is invalid.
    """
    self._compare_fn = compare_fn
    if self._compare_fn is None:
      raise ValueError('`compare_fn` must not be None.')
    _verify_compare_fn_args(self._compare_fn)

    self._saved_model_exporter = _SavedModelExporter(name,
                                                     serving_input_receiver_fn,
                                                     assets_extra, as_text)

    self._event_file_pattern = event_file_pattern
    self._model_dir = None
    self._best_eval_result = None

    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be a positive number. Got %s' %
          exports_to_keep)

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    export_result = None

    if self._model_dir != estimator.model_dir and self._event_file_pattern:
      # Loads best metric from event files.
      tf_logging.info('Loading best metric from event files.')

      self._model_dir = estimator.model_dir
      full_event_file_pattern = os.path.join(self._model_dir,
                                             self._event_file_pattern)
      self._best_eval_result = self._get_best_eval_result(
          full_event_file_pattern, eval_result)

    if self._best_eval_result is None or self._compare_fn(
        best_eval_result=self._best_eval_result,
        current_eval_result=eval_result):
      tf_logging.info('Performing best model export.')
      self._best_eval_result = eval_result
      export_result = self._saved_model_exporter.export(estimator, export_path,
                                                        checkpoint_path,
                                                        eval_result,
                                                        is_the_final_export)
      self._garbage_collect_exports(export_path)
      # cp best checkpoints to best folder
      model_dir, _ = os.path.split(checkpoint_path)
      # add / is to be compatiable with oss
      best_dir = os.path.join(model_dir, 'best_ckpt/')
      tf_logging.info('Copy best checkpoint %s to %s' %
                      (checkpoint_path, best_dir))
      if not gfile.Exists(best_dir):
        gfile.MakeDirs(best_dir)
      for tmp_file in gfile.Glob(checkpoint_path + '.*'):
        _, file_name = os.path.split(tmp_file)
        # skip temporary files
        if 'tempstate' in file_name:
          continue
        dst_path = os.path.join(best_dir, file_name)
        tf_logging.info('Copy file %s to %s' % (tmp_file, dst_path))
        try:
          gfile.Copy(tmp_file, dst_path)
        except Exception as ex:
          tf_logging.warn('Copy file %s to %s failed:  %s' %
                          (tmp_file, dst_path, str(ex)))
      self._garbage_collect_ckpts(best_dir)

    return export_result

  def _garbage_collect_ckpts(self, best_dir):
    """Deletes older best ckpts, retaining only a given number of the most recent.

    Args:
      best_dir: the directory where the n best ckpts are saved.
    """
    if self._exports_to_keep is None:
      return

    # delete older checkpoints
    tmp_files = gfile.Glob(os.path.join(best_dir, 'model.ckpt-*.meta'))
    if len(tmp_files) <= self._exports_to_keep:
      return

    tmp_steps = [_get_ckpt_version(x) for x in tmp_files]
    tmp_steps = sorted(tmp_steps)
    drop_num = len(tmp_steps) - self._exports_to_keep
    tf_logging.info(
        'garbage_collect_ckpts: steps: %s export_to_keep: %d drop num: %d' %
        (str(tmp_steps), self._exports_to_keep, drop_num))
    for ver in tmp_steps[:drop_num]:
      tmp_prefix = os.path.join(best_dir, 'model.ckpt-%d.*' % ver)
      for tmp_file in gfile.Glob(tmp_prefix):
        tf_logging.info('Remove ckpt file: ' + tmp_file)
        gfile.Remove(tmp_file)

  def _garbage_collect_exports(self, export_dir_base):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(io_util.fix_oss_dir(p.path))
      except errors_impl.NotFoundError as e:
        tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
    # pylint: enable=protected-access

  def _get_best_eval_result(self, event_files, curr_eval_result):
    """Get the best eval result from event files.

    Args:
      event_files: Absolute pattern of event files.

    Returns:
      The best eval result.
    """
    if not event_files:
      return None

    best_eval_result = None
    for event_file in gfile.Glob(os.path.join(event_files)):
      for event in summary_iterator.summary_iterator(event_file):
        if event.HasField('summary'):
          event_eval_result = {}
          event_eval_result['global_step'] = event.step
          if event.step >= curr_eval_result['global_step']:
            continue
          for value in event.summary.value:
            if value.HasField('simple_value'):
              event_eval_result[value.tag] = value.simple_value
          if len(event_eval_result) >= 2:
            if best_eval_result is None or self._compare_fn(
                best_eval_result, event_eval_result):
              best_eval_result = event_eval_result
    return best_eval_result


class FinalExporter(Exporter):
  """This class exports the serving graph and checkpoints at the end.

  This class performs a single export at the end of training.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._saved_model_exporter = _SavedModelExporter(name,
                                                     serving_input_receiver_fn,
                                                     assets_extra, as_text)

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    if not is_the_final_export:
      return None

    tf_logging.info('Performing the final export in the end of training.')

    return self._saved_model_exporter.export(estimator, export_path,
                                             checkpoint_path, eval_result,
                                             is_the_final_export)


class LatestExporter(Exporter):
  """This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False,
               exports_to_keep=5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._saved_model_exporter = _SavedModelExporter(name,
                                                     serving_input_receiver_fn,
                                                     assets_extra, as_text)
    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    export_result = self._saved_model_exporter.export(estimator, export_path,
                                                      checkpoint_path,
                                                      eval_result,
                                                      is_the_final_export)

    self._garbage_collect_exports(export_path)
    return export_result

  def _garbage_collect_exports(self, export_dir_base):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(io_util.fix_oss_dir(p.path))
      except errors_impl.NotFoundError as e:
        tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
    # pylint: enable=protected-access


class LargeExporter(Exporter):
  """This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               extra_params={},
               assets_extra=None,
               exports_to_keep=5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._name = name
    self._serving_input_fn = serving_input_receiver_fn
    self._assets_extra = assets_extra
    self._exports_to_keep = exports_to_keep
    self._extra_params = extra_params
    self._embedding_version = 0
    self._verbose = extra_params.get('verbose', False)
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    pipeline_config_path = os.path.join(estimator.model_dir, 'pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path)
    extra_params = dict(self._extra_params)
    # Exchange embedding_version to avoid conflict, such as the trainer is overwrite
    # embedding, while the online server is reading an old graph with the overwrited
    # embedding(which may be incomplete), so we use double versions to ensure stability.
    # Only two versions of embeddings are kept to reduce disk space consumption.
    extra_params['oss_path'] = os.path.join(extra_params['oss_path'],
                                            str(self._embedding_version))
    self._embedding_version = 1 - self._embedding_version
    if pipeline_config.train_config.HasField('incr_save_config'):
      incr_save_config = pipeline_config.train_config.incr_save_config
      extra_params['incr_update'] = {}
      incr_save_type = incr_save_config.WhichOneof('incr_update')
      logging.info('incr_save_type=%s' % incr_save_type)
      if incr_save_type:
        extra_params['incr_update'][incr_save_type] = getattr(
            incr_save_config, incr_save_type)
    else:
      incr_save_config = None
    export_result = export_big_model_to_oss(export_path, pipeline_config,
                                            extra_params,
                                            self._serving_input_fn, estimator,
                                            checkpoint_path, None,
                                            self._verbose)
    # clear old incr_save updates to reduce burden for file listing
    # at server side
    if incr_save_config is not None and incr_save_config.HasField('fs'):
      fs = incr_save_config.fs
      if fs.relative:
        incr_save_dir = os.path.join(estimator.model_dir, fs.incr_save_dir)
      else:
        incr_save_dir = fs.incr_save_dir
      global_step = int(checkpoint_path.split('-')[-1])
      limit_step = global_step - 1000
      if limit_step <= 0:
        limit_step = 0
      if limit_step > 0:
        dense_updates = gfile.Glob(os.path.join(incr_save_dir, 'dense_update*'))
        keep_ct, drop_ct = 0, 0
        for k in dense_updates:
          if not k.endswith('.done'):
            update_step = int(k.split('_')[-1])
            if update_step < limit_step:
              gfile.Remove(k + '.done')
              gfile.Remove(k)
              logging.info('clear old update: %s' % k)
              drop_ct += 1
            else:
              keep_ct += 1
        logging.info(
            '[global_step=%d][limit_step=%d] drop %d and keep %d dense_updates'
            % (global_step, limit_step, drop_ct, keep_ct))
        sparse_updates = gfile.Glob(
            os.path.join(incr_save_dir, 'sparse_update*'))
        keep_ct, drop_ct = 0, 0
        for k in sparse_updates:
          if not k.endswith('.done'):
            update_step = int(k.split('_')[-1])
            if update_step < limit_step:
              gfile.Remove(k + '.done')
              gfile.Remove(k)
              logging.info('clear old update: %s' % k)
              drop_ct += 1
            else:
              keep_ct += 1
        logging.info(
            '[global_step=%d][limit_step=%d] drop %d and keep %d sparse_updates'
            % (global_step, limit_step, drop_ct, keep_ct))
    self._garbage_collect_exports(export_path)
    return export_result

  def _garbage_collect_exports(self, export_dir_base):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(io_util.fix_oss_dir(p.path))
      except errors_impl.NotFoundError as e:
        logging.warning('Can not delete %s recursively: %s' % (p.path, e))
