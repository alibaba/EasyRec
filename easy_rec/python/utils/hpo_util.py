# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

import tensorflow as tf
from tensorflow.python.summary import summary_iterator

if tf.__version__ >= '2.0':
  gfile = tf.compat.v1.gfile
else:
  gfile = tf.gfile


def get_all_eval_result(event_file_pattern):
  """Get the best eval result from event files.

  Args:
    event_files: Absolute pattern of event files.

  Returns:
    The best eval result.
  """
  all_eval_result = []
  for event_file in gfile.Glob(os.path.join(event_file_pattern)):
    for event in summary_iterator.summary_iterator(event_file):
      if event.HasField('summary'):
        event_eval_result = {}
        event_eval_result['global_step'] = event.step
        for value in event.summary.value:
          if value.HasField('simple_value'):
            event_eval_result[value.tag] = value.simple_value
        if len(event_eval_result) >= 2:
          all_eval_result.append(event_eval_result)
  return all_eval_result


def save_eval_metrics(model_dir, metric_save_path, has_evaluator=True):

  def _get_eval_event_file_pattern():
    eval_dir = os.path.join(model_dir, 'eval_val/')
    if not gfile.Exists(eval_dir):
      eval_dir = os.path.join(model_dir, 'eval/')
      assert gfile.Exists(eval_dir), 'eval_val or eval does exists'
    event_file_pattern = os.path.join(eval_dir, '*.tfevents.*')
    logging.info('event_file_pattern: %s' % event_file_pattern)
    return event_file_pattern

  all_eval_res = []
  if 'TF_CONFIG' in os.environ:
    # check whether evaluator exists
    tf_config = json.loads(os.environ['TF_CONFIG'])
    logging.info('tf_config = %s' % json.dumps(tf_config))
    logging.info('model_dir = %s' % model_dir)
    if has_evaluator:
      if tf_config['task']['type'] == 'evaluator':
        event_file_pattern = _get_eval_event_file_pattern()
        all_eval_res = get_all_eval_result(event_file_pattern)
    elif 'master' in tf_config['cluster'] or 'chief' in tf_config['cluster']:
      if tf_config['task']['type'] in ['master', 'chief']:
        event_file_pattern = _get_eval_event_file_pattern()
        all_eval_res = get_all_eval_result(event_file_pattern)
    else:
      assert False, 'invalid cluster config, could not find master or chief or evaluator'
  else:
    # standalone mode
    event_file_pattern = _get_eval_event_file_pattern()
    all_eval_res = get_all_eval_result(event_file_pattern)

  logging.info('all_eval_res num = %d' % len(all_eval_res))
  if len(all_eval_res) > 0:
    with gfile.GFile(metric_save_path, 'w') as fout:
      for eval_res in all_eval_res:
        fout.write(json.dumps(eval_res) + '\n')
    logging.info('save all evaluation result to %s' % metric_save_path)


def kill_old_proc(tmp_dir, platform='pai'):
  old_proc_file = os.path.join(tmp_dir, 'old_proc.txt')
  if platform == 'pai':
    os.system(
        "ps -auxwww | grep easy_rec.python.hpo.pai_hpo | grep python | grep -v grep | awk '{ print $2 }' >> %s"
        % old_proc_file)
    os.system(
        "ps -auxwww | grep client/experiment_main.py | grep python | grep -v grep | awk '{ print $2 }' >> %s"
        % old_proc_file)
  else:
    os.system(
        "ps -auxwww | grep easy_rec.python.hpo.emr_hpo  | grep -v grep | awk '{ print $2 }' > %s"
        % old_proc_file)
    os.system(
        "ps -auxwww | grep client/experiment_main.py | grep python | grep -v grep | awk '{ print $2 }' >> %s "
        % old_proc_file)
    os.system(
        "ps -auxwww | grep el_submit | grep easy_rec_hpo | grep -v grep | awk '{ print $2 }' >> %s "
        % old_proc_file)
  proc_arr = []
  with open(old_proc_file, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      proc_arr.append(line_str)
  proc_arr = list(set(proc_arr))
  # remove current pid to avoid current process being killed
  pid = os.getpid()
  proc_arr.remove(str(pid))
  if len(proc_arr) > 0:
    logging.info('old process to be killed: %s' % ','.join(proc_arr))
    os.system('kill -9 %s' % (' '.join(proc_arr)))

  if platform == 'emr':
    # clear easy_rec_hpo yarn jobs
    yarn_job_file = os.path.join(tmp_dir, 'yarn_job.txt')
    os.system(
        "yarn application -list | awk '{ if ($2 == \"easy_rec_hpo\") print $1 }' > %s"
        % yarn_job_file)
    yarn_job_arr = []
    with open(yarn_job_file, 'r') as fin:
      for line_str in fin:
        line_str = line_str.strip()
        yarn_job_arr.append(line_str)
    yarn_job_arr = list(set(yarn_job_arr))
    if len(yarn_job_arr) > 0:
      logging.info('will kill the easy_rec_hpo yarn jobs: %s' %
                   ','.join(yarn_job_arr))
      os.system('yarn application -kill %s' % ' '.join(yarn_job_arr))
