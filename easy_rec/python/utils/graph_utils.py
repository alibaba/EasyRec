# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

from easy_rec.python.utils import pai_util

def graph_init(graph, tf_config=None):
  if tf_config:
    if isinstance(tf_config, str) or isinstance(tf_config, type(u'')):
      tf_config = json.loads(tf_config)
    if 'ps' in tf_config['cluster']:
      # ps mode
      logging.info('ps mode')
      ps_count = len(tf_config['cluster']['ps'])
      evaluator_cnt = 1
      # evaluator_cnt = 1 if pai_util.has_evaluator() else 0
      # if evaluator_cnt == 0:
      #   logging.warning(
      #       'evaluator is not set as an client in GraphLearn,'
      #       'if you actually set evaluator in TF_CONFIG, please do: export'
      #       ' HAS_EVALUATOR=1.')
      task_count = len(tf_config['cluster']['worker']) + 1 + evaluator_cnt
      cluster = {'server_count': ps_count, 'client_count': task_count}
      if tf_config['task']['type'] in ['chief', 'master']:
        graph.init(cluster=cluster, job_name='client', task_index=0)
      elif tf_config['task']['type'] == 'worker':
        graph.init(
            cluster=cluster,
            job_name='client',
            task_index=tf_config['task']['index'] + 2)
      # TODO(hongsheng.jhs): check cluster has evaluator or not?
      elif tf_config['task']['type'] == 'evaluator':
        graph.init(
            cluster=cluster,
            job_name='client',
            task_index=tf_config['task']['index'] + 1)
      elif tf_config['task']['type'] == 'ps':
        graph.init(
            cluster=cluster,
            job_name='server',
            task_index=tf_config['task']['index'])
    else:
      # worker mode
      logging.info('worker mode')
      task_count = len(tf_config['cluster']['worker']) + 1
      if tf_config['task']['type'] in ['chief', 'master']:
        graph.init(task_index=0, task_count=task_count)
      elif tf_config['task']['type'] == 'worker':
        graph.init(
            task_index=tf_config['task']['index'] + evaluator_cnt,
            task_count=task_count)
  else:
    # local mode
    graph.init()
