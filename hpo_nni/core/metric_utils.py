# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import pathlib
import re
import time
from threading import Thread

import nni
import oss2
from hpo_nni.core.utils import get_value
from hpo_nni.core.utils import set_value
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary_iterator


def judge_key(metric_dict, event_res):
  for key in metric_dict:
    if key not in event_res.keys():
      return False
  return True


def _get_best_eval_result(event_files,
                          metric_dict={'auc': 1},
                          trial_id=None,
                          use_best=False,
                          nni_report=True,
                          nni_report_final=False):
  if not event_files:
    return None

  best_eval_result = None
  best_event = None
  report_step = -1
  if trial_id:
    report_step = get_value(trial_id + '_report_step', -1, trial_id=trial_id)
  try:
    for event_file in gfile.Glob(os.path.join(event_files)):
      for event in summary_iterator.summary_iterator(event_file):
        if event.HasField('summary'):
          event_eval_result = {}
          event_eval_result['global_step'] = event.step
          for value in event.summary.value:
            if value.HasField('simple_value'):
              event_eval_result[value.tag] = value.simple_value

          if len(event_eval_result) >= 2 and judge_key(metric_dict,
                                                       event_eval_result):
            temp = 0
            for key in metric_dict:
              temp += metric_dict[key] * event_eval_result[key]

            if use_best:
              if best_eval_result is None or best_eval_result < temp:
                best_eval_result = temp
                best_event = event_eval_result
            else:  # use final result
              best_eval_result = temp
              best_event = event_eval_result
            if event.step > report_step and nni_report:
              nni.report_intermediate_result(temp)
              if trial_id:
                set_value(
                    trial_id + '_report_step', event.step, trial_id=trial_id)
              logging.info('event_eval_result: %s, temp metric: %s',
                           event_eval_result, temp)

  except Exception:
    logging.warning('the events is not ok,read the events error')

  if best_eval_result and nni_report and nni_report_final:
    nni.report_final_result(best_eval_result)

  return best_eval_result, best_event


def get_result(filepath,
               dst_filepath,
               metric_dict={'auc': 1},
               trial_id=None,
               oss_config=None,
               nni_report=True,
               use_best=False):
  if filepath:
    copy_dir(filepath, dst_filepath, oss_config)
  full_event_file_pattern = os.path.join(dst_filepath, '*.tfevents.*')
  logging.info('event_file: %s', full_event_file_pattern)
  best_eval_result, best_event = _get_best_eval_result(
      full_event_file_pattern,
      metric_dict=metric_dict,
      trial_id=trial_id,
      nni_report=nni_report,
      use_best=use_best)
  logging.info('best_metric: %s', best_eval_result)
  logging.info('best_event: %s', best_event)
  return best_eval_result, best_event


def get_bucket(ori_filepath, oss_config=None):
  # oss_config is the dict,such as:
  # {'accessKeyID':'xxx','accessKeySecret':'xxx','endpoint':'xxx'}
  auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
  cname = oss_config['endpoint']
  oss_pattern = re.compile(r'oss://([^/]+)/(.+)')
  m = oss_pattern.match(ori_filepath)
  if not m:
    raise IOError('invalid oss path: ' + ori_filepath +
                  ' should be oss://<bucket_name>/path')
  bucket_name, path = m.groups()
  path = path.replace('//', '/')
  bucket_name = bucket_name.split('.')[0]
  logging.info('bucket_name: %s, path: %s', bucket_name, path)

  bucket = oss2.Bucket(auth, cname, bucket_name)

  return bucket, path


def copy_dir(ori_filepath, dst_filepath, oss_config=None):
  logging.info('start copy from %s to %s', ori_filepath, dst_filepath)
  bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
  for b in oss2.ObjectIterator(bucket, path, delimiter='/'):
    if not b.is_prefix():
      file_name = b.key[b.key.rindex('/') + 1:]
      if len(file_name):
        pathlib.Path(dst_filepath).mkdir(parents=True, exist_ok=True)
        logging.info('downloadfile--> %s', b.key)
        bucket.get_object_to_file(b.key, os.path.join(dst_filepath, file_name))

  logging.info('copy end')


def copy_file(ori_filepath, dst_filepath, oss_config=None):
  logging.info('start copy from %s to %s', ori_filepath, dst_filepath)
  bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
  bucket.get_object_to_file(path, dst_filepath)


def upload_file(ori_filepath, dst_filepath, oss_config=None):
  logging.info('start upload to %s from %s', ori_filepath, dst_filepath)
  bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
  bucket.put_object_from_file(path, dst_filepath)


def report_result(filepath,
                  dst_filepath,
                  metric_dict,
                  trial_id=None,
                  oss_config=None,
                  nni_report=True,
                  use_best=False):
  worker = Thread(
      target=load_loop,
      args=(filepath, dst_filepath, metric_dict, trial_id, oss_config,
            nni_report, use_best))
  worker.start()


def load_loop(filepath, dst_filepath, metric_dict, trial_id, oss_config,
              nni_report, use_best):
  while True:
    best_eval_result, best_event = get_result(
        filepath,
        dst_filepath,
        metric_dict=metric_dict,
        trial_id=trial_id,
        oss_config=oss_config,
        nni_report=nni_report,
        use_best=use_best)
    # train end normaly
    if trial_id and get_value(trial_id + '_exit', trial_id=trial_id) == '1':
      if best_eval_result:
        nni.report_final_result(best_eval_result)
      logging.info('the job end')
      break
    time.sleep(30)
