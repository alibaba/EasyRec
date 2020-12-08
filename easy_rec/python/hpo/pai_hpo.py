# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hyperparameter search demo for easy_rec on pai."""
import json
import logging
import os
import shutil
import time

from pai.automl import hpo

from easy_rec.python.utils import hpo_util

file_dir, _ = os.path.split(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

try:
  import subprocess

  subprocess.check_output('which odpscmd', shell=True)
except Exception:
  logging.error(
      'odpscmd is not in path, please install from https://help.aliyun.com/document_detail/27971.html'
  )


def get_tuner(data, max_parallel, max_trial_num):
  param_dict = json.loads(data)
  if 'environment' in param_dict.keys():
    hpo.register_env(**param_dict['environment'])

  # hyper param
  params = []
  for h in param_dict['hyperparams']:
    param = hpo.hyperparam.create(**h)
    params.append(param)

  # tasks
  tasks = []
  for t in param_dict['tasks']:
    r = None
    if 'metric_reader' in t.keys():
      r = hpo.reader.create(**t['metric_reader'])
      t.pop('metric_reader')
    if r:
      subtask = hpo.task.create(**t, metric_reader=r)
    else:
      subtask = hpo.task.create(**t)
    tasks.append(subtask)

  # earlystop & algo
  early_stop = None
  if 'earlystop' in param_dict.keys():
    early_stop = hpo.earlystop.create(**param_dict['earlystop'])

  algo = None
  if 'algorithm' in param_dict.keys():
    algo = hpo.algorithm.create(**param_dict['algorithm'])

  tuner = hpo.autotuner.AutoTuner(
      earlystop=early_stop,
      algorithm=algo,
      hyperparams=params,
      task_list=tasks,
      max_parallel=max_parallel,
      max_trial_num=max_trial_num,
      mode='local',
      user_id='your_cloud_id')
  return tuner


def hpo_config(config_path, hyperparams, environment, exp_dir, tables, cluster,
               algo_proj_name, metric_name, odps_config_path):
  earlystop = {
      'type': 'large_is_better',
      'threshold': 0.99,
      'max_runtime': 2400
  }
  algorithm = {
      'type': 'gp',
      'initial_trials_num': 4,
      'stop_when_exception': True
  }

  if exp_dir.startswith('oss://'):
    exp_dir = exp_dir.replace('oss://', '')
    exp_dir = exp_dir[exp_dir.find('/') + 1:]

  param_path = '%s/hpo_test_{{ trial.id }}.json' % exp_dir
  metric_path = '%s/easy_rec_hpo_{{ trial.id }}.metric' % exp_dir
  model_path = '%s/easy_rec_hpo_{{ trial.id }}' % exp_dir
  bucket = 'oss://' + environment['bucket'].strip('/') + '/'

  adapter_task = {
      'type': 'ossadaptertask',
      # hpo_param_path for easy_rec
      'param_file': param_path,
  }

  tmp_dir = '/tmp/pai_easy_rec_hpo_%d' % time.time()
  os.makedirs(tmp_dir)
  logging.info('local temporary path: %s' % tmp_dir)

  def _add_prefix(table_name):
    table_name = table_name.strip()
    if not table_name.startswith('odps://'):
      return 'odps://%s/tables/%s' % (environment['project'], table_name)
    else:
      return table_name

  tables = [_add_prefix(x) for x in tables.split(',') if x != '']
  tables = ','.join(tables)
  logging.info('will tune on data: %s' % tables)

  sql_path = '%s/train_ext_hpo_{{ trial.id }}.sql' % tmp_dir
  prepare_sql_task = {
      'type':
          'BashTask',
      'cmd': [
          'python', '-m', 'easy_rec.python.hpo.generate_hpo_sql', '--sql_path',
          sql_path, '--config_path', config_path, '--tables', tables,
          '--cluster', cluster, '--bucket', bucket, '--hpo_param_path',
          os.path.join(bucket, param_path), '--hpo_metric_save_path',
          os.path.join(bucket, metric_path), '--model_dir',
          os.path.join(bucket, model_path), '--oss_host',
          environment['oss_endpoint'], '--role_arn', environment['role_arn'],
          '--algo_proj_name', algo_proj_name
      ]
  }

  train_task = {
      'type': 'BashTask',
      'cmd': ['odpscmd',
              '--config=%s' % odps_config_path, '-f', sql_path],
      'metric_reader': {
          'type': 'oss_reader',
          'location': metric_path,
          'parser_pattern': '.*"%s": (\\d.\\d+).*' % metric_name
      }
  }

  tasks = [adapter_task, prepare_sql_task, train_task]
  data = {
      'earlystop': earlystop,
      'algorithm': algorithm,
      'hyperparams': hyperparams,
      'tasks': tasks,
      'environment': environment
  }
  return data, tmp_dir


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--odps_config', type=str, help='odps_config.ini', default=None)
  parser.add_argument(
      '--oss_config', type=str, help='excel config path', default='')
  parser.add_argument('--bucket', type=str, help='bucket name', default=None)
  parser.add_argument('--role_arn', type=str, help='role arn', default=None)
  parser.add_argument(
      '--hyperparams', type=str, help='hyper parameters', default=None)
  parser.add_argument(
      '--config_path', type=str, help='pipeline config', default=None)
  parser.add_argument(
      '--tables', type=str, help='train table and test table', default=None)
  parser.add_argument(
      '--exp_dir', type=str, help='hpo experiment directory', default=None)
  parser.add_argument(
      '--cluster',
      type=str,
      help='cluster spec',
      default='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
  )
  parser.add_argument(
      '--algo_proj_name',
      type=str,
      help='algo project name',
      default='algo_public')
  parser.add_argument(
      '--metric_name', type=str, help='evaluate metric name', default='auc')
  parser.add_argument(
      '--max_parallel',
      type=int,
      help='max number of trials run at the same time',
      default=4)
  parser.add_argument(
      '--total_trial_num',
      type=int,
      help='total number of trials will run',
      default=6)
  parser.add_argument(
      '--debug',
      action='store_true',
      help='debug mode, will keep the temporary folder')

  args = parser.parse_args()

  assert os.path.exists(args.odps_config)
  odps_config = {}
  with open(args.odps_config, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      if len(line_str) == 0:
        continue
      if line_str[0] == '#':
        continue
      if '=' in line_str:
        tmp_id = line_str.find('=')
        key = line_str[:tmp_id].strip()
        val = line_str[(tmp_id + 1):].strip()
        odps_config[key] = val

  if args.oss_config is None:
    args.oss_config = os.path.join(os.environ['HOME'], '.ossutilconfig')
  assert os.path.exists(args.oss_config)
  oss_config = {}
  with open(args.oss_config, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      if len(line_str) == 0:
        continue
      if line_str[0] == '#':
        continue
      if '=' in line_str:
        tmp_id = line_str.find('=')
        key = line_str[:tmp_id].strip()
        val = line_str[(tmp_id + 1):].strip()
        oss_config[key] = val

  assert args.bucket is not None
  assert args.role_arn is not None

  if args.bucket.startswith('oss://'):
    args.bucket = args.bucket[len('oss://'):]

  environment = {
      'access_id': odps_config['access_id'],
      'access_key': odps_config['access_key'],
      'oss_access_id': oss_config['accessKeyID'],
      'oss_access_key': oss_config['accessKeySecret'],
      'project': odps_config['project_name'],
      'odps_endpoint': odps_config['end_point'],
      'biz_id': '147331^paistudio^xxxxxxx^2020-03-18',
      'role_arn': args.role_arn,
      'bucket': args.bucket,
      'oss_endpoint': oss_config['endpoint']
  }

  assert args.hyperparams is not None
  with open(args.hyperparams, 'r') as fin:
    hyperparams = json.load(fin)

  assert args.config_path is not None
  assert args.exp_dir is not None
  assert args.tables is not None

  data, tmp_dir = hpo_config(args.config_path, hyperparams, environment,
                             args.exp_dir, args.tables, args.cluster,
                             args.algo_proj_name, args.metric_name,
                             args.odps_config)
  hpo_util.kill_old_proc(tmp_dir, platform='pai')

  data_json = json.dumps(data)
  tuner = get_tuner(data_json, args.max_parallel, args.total_trial_num)
  tuner.fit(synchronize=True)

  if not args.debug:
    shutil.rmtree(tmp_dir)
  else:
    logging.info('temporary directory is: %s' % tmp_dir)
