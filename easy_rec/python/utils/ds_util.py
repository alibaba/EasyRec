# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import logging

def is_on_ds():
  # IS_ON_PAI is set in train_eval
  # which is the entry on DataScience platform
  return 'IS_ON_DS' in os.environ

def set_on_ds():
  logging.info('set on ds environment variable: IS_ON_DS')
  os.environ['IS_ON_DS'] = '1'
