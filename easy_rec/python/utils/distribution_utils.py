# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import json
import logging
import os

import tensorflow as tf

from easy_rec.python.protos.train_pb2 import DistributionStrategy
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils.estimator_utils import chief_to_master
from easy_rec.python.utils.estimator_utils import master_to_chief

DistributionStrategyMap = {
    '': DistributionStrategy.NoStrategy,
    'ps': DistributionStrategy.PSStrategy,
    'ess': DistributionStrategy.ExascaleStrategy,
    'mirrored': DistributionStrategy.MirroredStrategy,
    'collective': DistributionStrategy.CollectiveAllReduceStrategy
}


def set_distribution_config(pipeline_config, num_worker, num_gpus_per_worker,
                            distribute_strategy):
  if distribute_strategy in [
      DistributionStrategy.PSStrategy, DistributionStrategy.MirroredStrategy,
      DistributionStrategy.CollectiveAllReduceStrategy,
      DistributionStrategy.ExascaleStrategy
  ]:
    pipeline_config.train_config.sync_replicas = False
    pipeline_config.train_config.train_distribute = distribute_strategy
    pipeline_config.train_config.num_gpus_per_worker = num_gpus_per_worker
  print('Dump pipeline_config.train_config:')
  print(pipeline_config.train_config)


def set_tf_config_and_get_train_worker_num(
    ps_hosts,
    worker_hosts,
    task_index,
    job_name,
    distribute_strategy=DistributionStrategy.NoStrategy,
    eval_method='none'):
  logging.info(
      'set_tf_config_and_get_train_worker_num: distribute_strategy = %d' %
      distribute_strategy)
  worker_hosts = worker_hosts.split(',')
  ps_hosts = ps_hosts.split(',') if ps_hosts else []

  total_worker_num = len(worker_hosts)
  train_worker_num = total_worker_num

  print('Original TF_CONFIG=%s' % os.environ.get('TF_CONFIG', ''))
  print('worker_hosts=%s ps_hosts=%s task_index=%d job_name=%s' %
        (','.join(worker_hosts), ','.join(ps_hosts), task_index, job_name))
  print('eval_method=%s' % eval_method)

  if distribute_strategy == DistributionStrategy.MirroredStrategy:
    assert total_worker_num == 1, 'mirrored distribute strategy only need 1 worker'
  elif distribute_strategy in [
      DistributionStrategy.NoStrategy, DistributionStrategy.PSStrategy,
      DistributionStrategy.CollectiveAllReduceStrategy,
      DistributionStrategy.ExascaleStrategy
  ]:
    cluster, task_type, task_index_ = estimator_utils.parse_tf_config()
    train_worker_num = 0
    if eval_method == 'separate':
      if 'evaluator' in cluster:
        # 'evaluator' in cluster indicates user use new-style cluster content
        if 'chief' in cluster:
          train_worker_num += len(cluster['chief'])
        elif 'master' in cluster:
          train_worker_num += len(cluster['master'])
        if 'worker' in cluster:
          train_worker_num += len(cluster['worker'])
        # drop evaluator to avoid hang
        if distribute_strategy == DistributionStrategy.NoStrategy:
          del cluster['evaluator']
        tf_config = {
            'cluster': cluster,
            'task': {
                'type': task_type,
                'index': task_index_
            }
        }
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
      else:
        # backward compatibility, if user does not assign one evaluator in
        # -Dcluster, we use first worker for chief, second for evaluation
        train_worker_num = total_worker_num - 1
        assert train_worker_num > 0, 'in distribution mode worker num must be greater than 1, ' \
                                     'the second worker will be used as evaluator'
        if len(worker_hosts) > 1:
          cluster = {'chief': [worker_hosts[0]], 'worker': worker_hosts[2:]}
          if distribute_strategy != DistributionStrategy.NoStrategy:
            cluster['evaluator'] = [worker_hosts[1]]
          if len(ps_hosts) > 0:
            cluster['ps'] = ps_hosts
          if job_name == 'ps':
            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': cluster,
                'task': {
                    'type': job_name,
                    'index': task_index
                }
            })
          elif job_name == 'worker':
            if task_index == 0:
              os.environ['TF_CONFIG'] = json.dumps({
                  'cluster': cluster,
                  'task': {
                      'type': 'chief',
                      'index': 0
                  }
              })
            elif task_index == 1:
              os.environ['TF_CONFIG'] = json.dumps({
                  'cluster': cluster,
                  'task': {
                      'type': 'evaluator',
                      'index': 0
                  }
              })
            else:
              os.environ['TF_CONFIG'] = json.dumps({
                  'cluster': cluster,
                  'task': {
                      'type': job_name,
                      'index': task_index - 2
                  }
              })
    else:
      if 'evaluator' in cluster:
        evaluator = cluster['evaluator']
        del cluster['evaluator']
        # 'evaluator' in cluster indicates user use new-style cluster content
        train_worker_num += 1
        if 'chief' in cluster:
          train_worker_num += len(cluster['chief'])
        elif 'master' in cluster:
          train_worker_num += len(cluster['master'])
        if 'worker' in cluster:
          train_worker_num += len(cluster['worker'])
          cluster['worker'].append(evaluator[0])
        else:
          cluster['worker'] = [evaluator[0]]
        if task_type == 'evaluator':
          tf_config = {
              'cluster': cluster,
              'task': {
                  'type': 'worker',
                  'index': train_worker_num - 2
              }
          }
        else:
          tf_config = {
              'cluster': cluster,
              'task': {
                  'type': task_type,
                  'index': task_index_
              }
          }
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
      else:
        cluster = {'chief': [worker_hosts[0]], 'worker': worker_hosts[1:]}
        train_worker_num = len(worker_hosts)
        if len(ps_hosts) > 0:
          cluster['ps'] = ps_hosts
        if job_name == 'ps':
          os.environ['TF_CONFIG'] = json.dumps({
              'cluster': cluster,
              'task': {
                  'type': job_name,
                  'index': task_index
              }
          })
        else:
          if task_index == 0:
            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': cluster,
                'task': {
                    'type': 'chief',
                    'index': 0
                }
            })
          else:
            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': cluster,
                'task': {
                    'type': 'worker',
                    'index': task_index - 1
                }
            })
      if eval_method == 'none':
        # change master to chief, will not evaluate
        master_to_chief()
      elif eval_method == 'master':
        # change chief to master, will evaluate on master
        chief_to_master()
  else:
    assert distribute_strategy == '', 'invalid distribute_strategy %s'\
           % distribute_strategy
    cluster, task_type, task_index = estimator_utils.parse_tf_config()
  print('Final TF_CONFIG = %s' % os.environ.get('TF_CONFIG', ''))
  tf.logging.info('TF_CONFIG %s' % os.environ.get('TF_CONFIG', ''))
  tf.logging.info('distribute_stategy %s, train_worker_num: %d' %
                  (distribute_strategy, train_worker_num))

  # remove pai chief-worker waiting strategy
  # which is conflicted with worker waiting strategy in easyrec
  if 'TF_WRITE_WORKER_STATUS_FILE' in os.environ:
    del os.environ['TF_WRITE_WORKER_STATUS_FILE']
  return train_worker_num


def set_tf_config_and_get_train_worker_num_on_ds():
  if 'TF_CONFIG' not in os.environ:
    return
  tf_config = json.loads(os.environ['TF_CONFIG'])
  if 'cluster' in tf_config and 'ps' in tf_config['cluster'] and (
      'evaluator' not in tf_config['cluster']):
    easyrec_tf_config = dict()
    easyrec_tf_config['cluster'] = {}
    easyrec_tf_config['task'] = {}
    easyrec_tf_config['cluster']['ps'] = tf_config['cluster']['ps']
    easyrec_tf_config['cluster']['chief'] = [tf_config['cluster']['worker'][0]]
    easyrec_tf_config['cluster']['worker'] = tf_config['cluster']['worker'][2:]

    if tf_config['task']['type'] == 'worker' and tf_config['task']['index'] == 0:
      easyrec_tf_config['task']['type'] = 'chief'
      easyrec_tf_config['task']['index'] = 0
    elif tf_config['task']['type'] == 'worker' and tf_config['task'][
        'index'] == 1:
      easyrec_tf_config['task']['type'] = 'evaluator'
      easyrec_tf_config['task']['index'] = 0
    elif tf_config['task']['type'] == 'worker':
      easyrec_tf_config['task']['type'] = tf_config['task']['type']
      easyrec_tf_config['task']['index'] = tf_config['task']['index'] - 2
    else:
      easyrec_tf_config['task']['type'] = tf_config['task']['type']
      easyrec_tf_config['task']['index'] = tf_config['task']['index']
    os.environ['TF_CONFIG'] = json.dumps(easyrec_tf_config)


def set_tf_config_and_get_distribute_eval_worker_num_on_ds():
  assert 'TF_CONFIG' in os.environ, "'TF_CONFIG' must in os.environ"
  tf_config = json.loads(os.environ['TF_CONFIG'])
  if 'cluster' in tf_config and 'ps' in tf_config['cluster'] and (
      'evaluator' not in tf_config['cluster']):
    easyrec_tf_config = dict()
    easyrec_tf_config['cluster'] = {}
    easyrec_tf_config['task'] = {}
    easyrec_tf_config['cluster']['ps'] = tf_config['cluster']['ps']
    easyrec_tf_config['cluster']['chief'] = [tf_config['cluster']['worker'][0]]
    easyrec_tf_config['cluster']['worker'] = tf_config['cluster']['worker'][1:]

    if tf_config['task']['type'] == 'worker' and tf_config['task']['index'] == 0:
      easyrec_tf_config['task']['type'] = 'chief'
      easyrec_tf_config['task']['index'] = 0
    elif tf_config['task']['type'] == 'worker':
      easyrec_tf_config['task']['type'] = tf_config['task']['type']
      easyrec_tf_config['task']['index'] = tf_config['task']['index'] - 1
    else:
      easyrec_tf_config['task']['type'] = tf_config['task']['type']
      easyrec_tf_config['task']['index'] = tf_config['task']['index']
    os.environ['TF_CONFIG'] = json.dumps(easyrec_tf_config)
