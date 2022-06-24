import os
import pathlib
import re
import time
from threading import Thread

import nni
import oss2
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary_iterator

from easy_rec.python.hpo_nni.pai_nni.code.utils import get_value
from easy_rec.python.hpo_nni.pai_nni.code.utils import parse_config
from easy_rec.python.hpo_nni.pai_nni.code.utils import set_value


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
      print(event_file)
      for event in summary_iterator.summary_iterator(event_file):
        if event.HasField('summary'):
          event_eval_result = {}
          event_eval_result['global_step'] = event.step
          for value in event.summary.value:
            if value.HasField('simple_value'):
              event_eval_result[value.tag] = value.simple_value

          if len(event_eval_result) >= 2:
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
              print('event_eval_result:', event_eval_result, ' temp metric:',
                    temp)

  except Exception as exception:
    print('the events is not ok,read the events error')
    print('exception:', exception)
  finally:
    print('read end')

  if best_eval_result and nni_report and nni_report_final:
    nni.report_final_result(best_eval_result)

  return best_eval_result, best_event


def get_result(filepath,
               dst_filepath,
               metric_dict={'auc': 1},
               trial_id=None,
               oss_config=None,
               nni_report=True):
  if filepath:
    copy_dir(filepath, dst_filepath, oss_config)
  full_event_file_pattern = os.path.join(dst_filepath, '*.tfevents.*')
  print('event_file:', full_event_file_pattern)
  best_eval_result, best_event = _get_best_eval_result(
      full_event_file_pattern,
      metric_dict=metric_dict,
      trial_id=trial_id,
      nni_report=nni_report)
  print('best_metric:', best_eval_result)
  print('best event:', best_event)
  return best_eval_result, best_event


def get_bucket(ori_filepath, oss_config=None):
  if not oss_config:
    oss_config = os.path.join(os.environ['HOME'], '.ossutilconfig')
  oss_config = parse_config(oss_config)
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
  print(bucket_name, path)

  bucket = oss2.Bucket(auth, cname, bucket_name)

  return bucket, path


def copy_dir(ori_filepath, dst_filepath, oss_config=None):
  print('start copy')
  bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
  for b in oss2.ObjectIterator(bucket, path, delimiter='/'):
    if not b.is_prefix():
      file_name = b.key[b.key.rindex('/') + 1:]
      if len(file_name):
        pathlib.Path(dst_filepath).mkdir(parents=True, exist_ok=True)
        print('downloadfile-->', b.key)
        bucket.get_object_to_file(b.key, os.path.join(dst_filepath, file_name))

  print('copy end')


def copy_file(ori_filepath, dst_filepath, oss_config=None):
  print('start copy')
  bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
  bucket.get_object_to_file(path, dst_filepath)


def upload_file(ori_filepath, dst_filepath, oss_config=None):
  print('start copy')
  bucket, path = get_bucket(ori_filepath=ori_filepath, oss_config=oss_config)
  bucket.put_object_from_file(path, dst_filepath)


def report_result(filepath,
                  dst_filepath,
                  metric_dict,
                  trial_id=None,
                  oss_config=None,
                  nni_report=True):
  worker = Thread(
      target=load_loop,
      args=(filepath, dst_filepath, metric_dict, trial_id, oss_config,
            nni_report))
  worker.start()


def load_loop(filepath, dst_filepath, metric_dict, trial_id, oss_config,
              nni_report):
  while True:
    print('get result')
    best_eval_result, best_event = get_result(
        filepath,
        dst_filepath,
        metric_dict=metric_dict,
        trial_id=trial_id,
        oss_config=oss_config,
        nni_report=nni_report)
    print(
        get_value(trial_id + '_exit', trial_id=trial_id),
        get_value(trial_id + '_exit', trial_id=trial_id) == '1')
    # train end normaly
    if trial_id and get_value(trial_id + '_exit', trial_id=trial_id) == '1':
      if best_eval_result:
        nni.report_final_result(best_eval_result)
      print('the job end')
      break
    time.sleep(30)
