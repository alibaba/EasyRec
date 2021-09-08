# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import numpy as np
import json

def custom_early_stop_func(eval_results, func_param):
  params = json.loads(func_param)
  tmp_thre = params['thre']
  metric_name = params['metric_name']
  for step, metrics in eval_results.items():
    val = metrics[metric_name]
    if val > tmp_thre:
      logging.info('At step %s, metric "%s" has value %s, which is larger than %s, causing early stop.' % (step, metric_name, val, tmp_thre))
      return True
  return False 
