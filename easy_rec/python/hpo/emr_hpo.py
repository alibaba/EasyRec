# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hyperparameter search for easy_rec on emr."""
import argparse
import json
import logging
import os
import shutil
import time

from pai.automl.hpo.autotuner import AutoTuner

from easy_rec.python.utils import hpo_util

file_dir, _ = os.path.split(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def hpo_config(config_path, hyperparams, exp_dir, metric_name,
               el_submit_params):
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

  tmp_dir = '/tmp/emr_easy_rec_hpo_%d' % time.time()
  os.makedirs(tmp_dir)
  logging.info('local temporary path: %s' % tmp_dir)

  param_path = tmp_dir + '/rewrite_{{ trial.id }}.json'
  param_path_file = 'rewrite_{{ trial.id }}.json'
  model_path = '%s/trail_{{ trial.id }}' % exp_dir
  metric_path = os.path.join(model_path, 'res.metric')

  pre_task = {
      'type': 'BashTask',
      'cmd': ['hadoop', 'fs', '-mkdir', '-p', model_path]
  }
  adapter_task = {
      'type': 'localadaptertask',
      # hpo_param_path for easy_rec
      'param_file': param_path,
  }
  el_params = [
      x.strip() for x in el_submit_params.split(' ') if x.strip() != ''
  ]
  assert len(
      el_params) % 2 == 0, 'invalid number of el_submit params: %d[%s]' % (
          len(el_params), str(el_params))
  for i in range(0, len(el_params), 2):
    assert el_params[i] in [
        '-t', '-m', '-pn', '-pc', '-pg', '-pm', '-wn', '-wc', '-wm', '-wg'
    ]
  cmd = ['el_submit'] + el_params + [
      '-a', 'easy_rec_hpo', '-m', 'local', '-f', '{},train_eval.py,{}'.format(
          config_path, param_path), '--interact', 'INTERACT', '-c',
      'python -m easy_rec.python.train_eval --hpo_metric_save_path {} '
      '--hpo_param_path {} --pipeline_config_path {} --model_dir {}'.format(
          metric_path, param_path_file, config_path, model_path)
  ]

  train_task = {
      'type': 'BashTask',
      'cmd': cmd,
      'metric_reader': {
          'type': 'hdfs_reader',
          'location': metric_path,
          'parser_pattern': '.*"%s": (\\d.\\d+).*' % metric_name
      }
  }

  tasks = [pre_task, adapter_task, train_task]
  data = {
      'earlystop': earlystop,
      'algorithm': algorithm,
      'hyperparams': hyperparams,
      'tasks': tasks
  }
  return data, tmp_dir


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--hyperparams', type=str, help='hyper parameters', default=None)
  parser.add_argument(
      '--config_path', type=str, help='pipeline config', default=None)
  parser.add_argument(
      '--exp_dir', type=str, help='hpo experiment directory', default=None)
  parser.add_argument(
      '--el_submit_params',
      type=str,
      help='el_submit parameters(-t x -m x [-pn x -pc x -pm x] -wn x -wc x -wm x -wg x)',
      default='-t standalone -m local -wn 1 -wc 6 -wm 20000 -wg 1')
  parser.add_argument(
      '--metric_name', type=str, help='metric_name', default='auc')
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

  assert args.hyperparams is not None
  assert args.config_path is not None
  assert args.exp_dir is not None

  with open(args.hyperparams, 'r') as fin:
    hyperparams = json.load(fin)

  data, tmp_dir = hpo_config(args.config_path, hyperparams, args.exp_dir,
                             args.metric_name, args.el_submit_params)

  hpo_util.kill_old_proc(tmp_dir, platform='emr')

  tuner = AutoTuner.create_tuner(
      data, max_parallel=args.max_parallel, max_trial_num=args.total_trial_num)
  tuner.fit(synchronize=True)

  if not args.debug:
    shutil.rmtree(tmp_dir)
  else:
    logging.info('temporary directory is: %s' % tmp_dir)
