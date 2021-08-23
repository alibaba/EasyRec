# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import sys
import traceback

import tensorflow as tf

if sys.version_info.major == 2:
  from urllib2 import urlopen, Request, HTTPError
else:
  from urllib.request import urlopen, Request
  from urllib.error import HTTPError


def is_on_pai():
  # IS_ON_PAI is set in pai_jobs/run.py
  # which is the entry on pai platform
  return 'IS_ON_PAI' in os.environ


def set_on_pai():
  logging.info('set on pai environment variable: IS_ON_PAI')
  os.environ['IS_ON_PAI'] = '1'


def download(url):
  _, fname = os.path.split(url)
  request = Request(url=url)
  try:
    response = urlopen(request, timeout=10)
    with open(fname, 'w') as ofile:
      ofile.write(response.read())
    return fname
  except HTTPError as e:
    tf.logging.error('http error: ', e.code)
    tf.logging.error('body:', e.read())
    return None
  except Exception as e:
    tf.logging.error(e)
    tf.logging.error(traceback.format_exc())
    return None


def process_config(configs, task_index=0, worker_num=1):
  """Download config and select config for the worker.

  Args:
    configs: config paths, separated by ','
    task_index: worker index
    worker_num: total number of workers
  """
  configs = configs.split(',')
  if len(configs) > 1:
    assert len(configs) == worker_num, \
        'number of configs must be equal to number of workers,' + \
        ' when number of configs > 1'
    config = configs[task_index]
  else:
    config = configs[0]

  if config[:4] == 'http':
    return download(config)
  elif config[:3] == 'oss':
    if '/##/' in config:
      config = config.replace('/##/', '\x02')
    if '/#/' in config:
      config = config.replace('/#/', '\x01')
    return config
  else:
    # allow to use this entry file to run experiments from local env
    # to avoid uploading sample file
    return config


def test():
  f = download(
      'https://easy-rec.oss-cn-hangzhou.aliyuncs.com/config/MultiTower/dwd_avazu_ctr_deepmodel.config'
  )
  assert f == 'dwd_avazu_ctr_deepmodel.config'


if __name__ == '__main__':
  test()
