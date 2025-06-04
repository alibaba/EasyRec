# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn import metrics as sklearn_metrics
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

from easy_rec.python.utils.estimator_utils import get_task_index_and_num
from easy_rec.python.utils.io_util import read_data_from_json_path
from easy_rec.python.utils.io_util import save_data_to_json_path
from easy_rec.python.utils.shape_utils import get_shape_list

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def max_f1(label, predictions):
  """Calculate the largest F1 metric under different thresholds.

  Args:
    label: Ground truth (correct) target values.
    predictions: Estimated targets as returned by a model.
  """
  from easy_rec.python.core.easyrec_metrics import metrics_tf
  num_thresholds = 200
  kepsilon = 1e-7
  thresholds = [
      (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
  ]
  thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

  f1_scores = []
  precision_update_ops = []
  recall_update_ops = []
  for threshold in thresholds:
    pred = (predictions > threshold)
    precision, precision_update_op = metrics_tf.precision(
        labels=label, predictions=pred, name='precision_%s' % threshold)
    recall, recall_update_op = metrics_tf.recall(
        labels=label, predictions=pred, name='recall_%s' % threshold)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-12)
    precision_update_ops.append(precision_update_op)
    recall_update_ops.append(recall_update_op)
    f1_scores.append(f1_score)

  f1 = tf.math.reduce_max(tf.stack(f1_scores))
  f1_update_op = tf.group(precision_update_ops + recall_update_ops)
  return f1, f1_update_op


def _separated_auc_impl(labels, predictions, keys, reduction='mean'):
  """Computes the AUC group by the key separately.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    keys: keys to be group by, A int or string `Tensor` whose shape matches `predictions`.
    reduction: reduction metric for auc of different keys
      * "mean": simple mean of different keys
      * "mean_by_sample_num": weighted mean with sample num of different keys
      * "mean_by_positive_num": weighted mean with positive sample num of different keys
  """
  assert reduction in ['mean', 'mean_by_sample_num', 'mean_by_positive_num'], \
      'reduction method must in mean | mean_by_sample_num | mean_by_positive_num'
  separated_label = defaultdict(list)
  separated_prediction = defaultdict(list)
  separated_weights = defaultdict(int)

  def update_pyfunc(labels, predictions, keys):
    for label, prediction, key in zip(labels, predictions, keys):
      separated_label[key].append(label)
      separated_prediction[key].append(prediction)
      if reduction == 'mean':
        separated_weights[key] = 1
      elif reduction == 'mean_by_sample_num':
        separated_weights[key] += 1
      elif reduction == 'mean_by_positive_num':
        separated_weights[key] += label

  def value_pyfunc():
    metrics = []
    weights = []
    for key in separated_label.keys():
      per_label = np.asarray(separated_label[key]).reshape([-1])
      per_prediction = np.asarray(separated_prediction[key]).reshape([-1])
      if np.all(per_label == 1) or np.all(per_label == 0):
        continue
      metric = sklearn_metrics.roc_auc_score(per_label, per_prediction)
      metrics.append(metric)
      weights.append(separated_weights[key])
    if len(metrics) > 0:
      return np.average(metrics, weights=weights).astype(np.float32)
    else:
      return np.float32(0.0)

  update_op = tf.py_func(update_pyfunc, [labels, predictions, keys], [])
  value_op = tf.py_func(value_pyfunc, [], tf.float32)
  return value_op, update_op


def fast_auc(labels, predictions, name, num_thresholds=1e5):
  num_thresholds = int(num_thresholds)

  def value_pyfunc(pos_neg_arr, total_pos_neg):
    partial_sum_pos = 0
    auc = 0
    total_neg = total_pos_neg[0]
    total_pos = total_pos_neg[1]
    for i in range(num_thresholds + 1):
      partial_sum_pos += pos_neg_arr[1][i]
      auc += (total_pos - partial_sum_pos) * pos_neg_arr[0][i] * 2
      auc += pos_neg_arr[0][i] * pos_neg_arr[1][i]
    auc = np.double(auc) / np.double(total_pos * total_neg * 2)
    logging.info('fast_auc[%s]: total_pos=%d total_neg=%d total=%d' %
                 (name, total_pos, total_neg, total_pos + total_neg))
    return np.float32(auc)

  with variable_scope.variable_scope(name_or_scope=name), tf.name_scope(name):
    neg_pos_var = variable_scope.get_variable(
        name='neg_pos_cnt',
        shape=[2, num_thresholds + 1],
        trainable=False,
        collections=[tf.GraphKeys.METRIC_VARIABLES],
        initializer=tf.zeros_initializer(),
        dtype=tf.int64)
    total_var = variable_scope.get_variable(
        name='total_cnt',
        shape=[2],
        trainable=False,
        collections=[tf.GraphKeys.METRIC_VARIABLES],
        initializer=tf.zeros_initializer(),
        dtype=tf.int64)
    pred_bins = math_ops.cast(predictions * num_thresholds, dtype=tf.int32)
    labels = math_ops.cast(labels, dtype=tf.int32)
    labels = array_ops.reshape(labels, [-1, 1])
    pred_bins = array_ops.reshape(pred_bins, [-1, 1])
    update_op0 = state_ops.scatter_nd_add(
        neg_pos_var, tf.concat([labels, pred_bins], axis=1),
        array_ops.ones(tf.shape(labels)[0], dtype=tf.int64))
    total_pos = math_ops.reduce_sum(labels)
    total_neg = array_ops.shape(labels)[0] - total_pos
    total_add = math_ops.cast(tf.stack([total_neg, total_pos]), dtype=tf.int64)
    update_op1 = state_ops.assign_add(total_var, total_add)
    return tf.py_func(value_pyfunc, [neg_pos_var, total_var],
                      tf.float32), tf.group([update_op0, update_op1])


def _distribute_separated_auc_impl(labels,
                                   predictions,
                                   keys,
                                   reduction='mean',
                                   metric_name='sepatated_auc'):
  """Computes the AUC group by the key separately.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    keys: keys to be group by, A int or string `Tensor` whose shape matches `predictions`.
    reduction: reduction metric for auc of different keys
    metric_name: the name of compute metric
      * "mean": simple mean of different keys
      * "mean_by_sample_num": weighted mean with sample num of different keys
      * "mean_by_positive_num": weighted mean with positive sample num of different keys
  """
  assert reduction in ['mean', 'mean_by_sample_num', 'mean_by_positive_num'], \
      'reduction method must in mean | mean_by_sample_num | mean_by_positive_num'
  separated_label = defaultdict(list)
  separated_prediction = defaultdict(list)
  separated_weights = defaultdict(int)
  tf_config = json.loads(os.environ['TF_CONFIG'])
  cur_job_name = tf_config['task']['type']
  cur_task_index, task_num = get_task_index_and_num()
  cur_work_device = 'job_' + cur_job_name + '__' + 'task_' + str(cur_task_index)
  eval_tmp_results_dir = os.environ['eval_tmp_results_dir']
  assert tf.gfile.IsDirectory(
      eval_tmp_results_dir), 'eval_tmp_results_dir not exists'

  def update_pyfunc(labels, predictions, keys):
    for label, prediction, key in zip(labels, predictions, keys):
      key = str(key)
      separated_label[key].append(label.item())
      separated_prediction[key].append(prediction.item())
      if reduction == 'mean':
        separated_weights[key] = 1
      elif reduction == 'mean_by_sample_num':
        separated_weights[key] += 1
      elif reduction == 'mean_by_positive_num':
        separated_weights[key] += label.item()
    for name, data in zip(
        ['separated_label', 'separated_prediction', 'separated_weights'],
        [separated_label, separated_prediction, separated_weights]):
      cur_json_name = metric_name + '__' + cur_work_device + '__' + name + '.json'
      cur_json_path = os.path.join(eval_tmp_results_dir, cur_json_name)
      save_data_to_json_path(cur_json_path, data)

  def value_pyfunc():
    for task_i in range(1, task_num):
      work_device_i = 'job_worker__task_' + str(task_i)
      for name in [
          'separated_label', 'separated_prediction', 'separated_weights'
      ]:
        json_name_i = metric_name + '__' + work_device_i + '__' + name + '.json'
        json_path_i = os.path.join(eval_tmp_results_dir, json_name_i)
        data_i = read_data_from_json_path(json_path_i)
        if (name == 'separated_label'):
          separated_label.update({
              key: separated_label.get(key, []) + data_i.get(key, [])
              for key in set(
                  list(separated_label.keys()) + list(data_i.keys()))
          })
        elif (name == 'separated_prediction'):
          separated_prediction.update({
              key: separated_prediction.get(key, []) + data_i.get(key, [])
              for key in set(
                  list(separated_prediction.keys()) + list(data_i.keys()))
          })
        elif (name == 'separated_weights'):
          if reduction == 'mean':
            separated_weights.update(data_i)
          else:
            separated_weights.update({
                key: separated_weights.get(key, 0) + data_i.get(key, 0)
                for key in set(
                    list(separated_weights.keys()) + list(data_i.keys()))
            })
        else:
          assert False, 'Not supported name {}'.format(name)
    metrics = []
    weights = []
    for key in separated_label.keys():
      per_label = np.asarray(separated_label[key]).reshape([-1])
      per_prediction = np.asarray(separated_prediction[key]).reshape([-1])
      if np.all(per_label == 1) or np.all(per_label == 0):
        continue
      metric = sklearn_metrics.roc_auc_score(per_label, per_prediction)
      metrics.append(metric)
      weights.append(separated_weights[key])
    if len(metrics) > 0:
      return np.average(metrics, weights=weights).astype(np.float32)
    else:
      return np.float32(0.0)

  update_op = tf.py_func(update_pyfunc, [labels, predictions, keys], [])
  value_op = tf.py_func(value_pyfunc, [], tf.float32)
  return value_op, update_op


def gauc(labels, predictions, uids, reduction='mean'):
  """Computes the AUC group by user separately.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    uids: user ids, A int or string `Tensor` whose shape matches `predictions`.
    reduction: reduction method for auc of different users
      * "mean": simple mean of different users
      * "mean_by_sample_num": weighted mean with sample num of different users
      * "mean_by_positive_num": weighted mean with positive sample num of different users
  """
  if os.environ.get('distribute_eval') == 'True':
    return _distribute_separated_auc_impl(
        labels, predictions, uids, reduction, metric_name='gauc')
  return _separated_auc_impl(labels, predictions, uids, reduction)


def session_auc(labels, predictions, session_ids, reduction='mean'):
  """Computes the AUC group by session separately.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    session_ids: session ids, A int or string `Tensor` whose shape matches `predictions`.
    reduction: reduction method for auc of different sessions
      * "mean": simple mean of different sessions
      * "mean_by_sample_num": weighted mean with sample num of different sessions
      * "mean_by_positive_num": weighted mean with positive sample num of different sessions
  """
  if os.environ.get('distribute_eval') == 'True':
    return _distribute_separated_auc_impl(
        labels, predictions, session_ids, reduction, metric_name='session_auc')
  return _separated_auc_impl(labels, predictions, session_ids, reduction)


def metric_learning_recall_at_k(k,
                                embeddings,
                                labels,
                                session_ids=None,
                                embed_normed=False):
  """Computes the recall_at_k metric for metric learning.

  Args:
    k: a scalar of int, or a tuple of ints
    embeddings: the output of last hidden layer, a tf.float32 `Tensor` with shape [batch_size, embedding_size]
    labels: a `Tensor` with shape [batch_size]
    session_ids: session ids, a `Tensor` with shape [batch_size]
    embed_normed: indicator of whether the input embeddings are l2_normalized
  """
  from easy_rec.python.core.easyrec_metrics import metrics_tf
  # make sure embedding should be l2-normalized
  if not embed_normed:
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
  embed_shape = get_shape_list(embeddings)
  batch_size = embed_shape[0]
  sim_mat = tf.matmul(embeddings, embeddings, transpose_b=True)
  sim_mat = sim_mat - tf.eye(batch_size) * 2.0
  indices_not_equal = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
  # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
  labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
  if session_ids is not None and session_ids is not labels:
    sessions_equal = tf.equal(
        tf.expand_dims(session_ids, 0), tf.expand_dims(session_ids, 1))
    labels_equal = tf.logical_and(sessions_equal, labels_equal)
  mask = tf.logical_and(indices_not_equal, labels_equal)
  mask_pos = tf.where(
      mask, sim_mat,
      -array_ops.ones_like(sim_mat))  # shape: (batch_size, batch_size)
  if isinstance(k, int):
    _, pos_top_k_idx = tf.nn.top_k(mask_pos, k)  # shape: (batch_size, k)
    return metrics_tf.recall_at_k(
        labels=tf.to_int64(pos_top_k_idx), predictions=sim_mat, k=k)
  if any((isinstance(k, list), isinstance(k, tuple), isinstance(k, set))):
    metrics = {}
    for kk in k:
      if kk < 1:
        continue
      _, pos_top_k_idx = tf.nn.top_k(mask_pos, kk)
      metrics['recall@' + str(kk)] = metrics_tf.recall_at_k(
          labels=tf.to_int64(pos_top_k_idx), predictions=sim_mat, k=kk)
    return metrics
  else:
    raise ValueError('k should be a `int` or a list/tuple/set of int.')


def metric_learning_average_precision_at_k(k,
                                           embeddings,
                                           labels,
                                           session_ids=None,
                                           embed_normed=False):
  from easy_rec.python.core.easyrec_metrics import metrics_tf
  # make sure embedding should be l2-normalized
  if not embed_normed:
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
  embed_shape = get_shape_list(embeddings)
  batch_size = embed_shape[0]
  sim_mat = tf.matmul(embeddings, embeddings, transpose_b=True)
  sim_mat = sim_mat - tf.eye(batch_size) * 2.0
  mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
  if session_ids is not None and session_ids is not labels:
    sessions_equal = tf.equal(
        tf.expand_dims(session_ids, 0), tf.expand_dims(session_ids, 1))
    mask = tf.logical_and(sessions_equal, mask)
  label_indices = _get_matrix_mask_indices(mask)
  if isinstance(k, int):
    return metrics_tf.average_precision_at_k(label_indices, sim_mat, k)
  if any((isinstance(k, list), isinstance(k, tuple), isinstance(k, set))):
    metrics = {}
    for kk in k:
      if kk < 1:
        continue
      metrics['MAP@' + str(kk)] = metrics_tf.average_precision_at_k(
          label_indices, sim_mat, kk)
    return metrics
  else:
    raise ValueError('k should be a `int` or a list/tuple/set of int.')


def _get_matrix_mask_indices(matrix, num_rows=None):
  if num_rows is None:
    num_rows = get_shape_list(matrix)[0]
  indices = tf.where(matrix)
  num_indices = tf.shape(indices)[0]
  elem_per_row = tf.bincount(
      tf.cast(indices[:, 0], tf.int32), minlength=num_rows)
  max_elem_per_row = tf.reduce_max(elem_per_row)
  row_start = tf.concat([[0], tf.cumsum(elem_per_row[:-1])], axis=0)
  r = tf.range(max_elem_per_row)
  idx = tf.expand_dims(row_start, 1) + r
  idx = tf.minimum(idx, num_indices - 1)
  result = tf.gather(indices[:, 1], idx)
  # replace invalid elements with -1
  result = tf.where(
      tf.expand_dims(elem_per_row, 1) > r, result, -array_ops.ones_like(result))
  max_index_per_row = tf.reduce_max(result, axis=1, keepdims=True)
  max_index_per_row = tf.tile(max_index_per_row, [1, max_elem_per_row])
  result = tf.where(result >= 0, result, max_index_per_row)
  return result
