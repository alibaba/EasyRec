# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import subprocess
import traceback

from tensorflow.python.platform import gfile

from easy_rec.python.utils import estimator_utils


def is_on_ds():
  # IS_ON_PAI is set in train_eval
  # which is the entry on DataScience platform
  return 'IS_ON_DS' in os.environ


def set_on_ds():
  logging.info('set on ds environment variable: IS_ON_DS')
  os.environ['IS_ON_DS'] = '1'


def cache_ckpt(pipeline_config):
  fine_tune_ckpt_path = pipeline_config.train_config.fine_tune_checkpoint
  if not fine_tune_ckpt_path.startswith('hdfs://'):
    # there is no need to cache if remote directories are mounted
    return

  if estimator_utils.is_ps() or estimator_utils.is_chief(
  ) or estimator_utils.is_master():
    tmpdir = os.path.dirname(fine_tune_ckpt_path.replace('hdfs://', ''))
    tmpdir = os.path.join('/tmp/experiments', tmpdir)
    logging.info('will cache fine_tune_ckpt to local dir: %s' % tmpdir)
    if gfile.IsDirectory(tmpdir):
      gfile.DeleteRecursively(tmpdir)
    gfile.MakeDirs(tmpdir)
    src_files = gfile.Glob(fine_tune_ckpt_path + '*')
    src_files.sort()
    data_files = [x for x in src_files if '.data-' in x]
    meta_files = [x for x in src_files if '.data-' not in x]
    if estimator_utils.is_ps():
      _, _, ps_id = estimator_utils.parse_tf_config()
      ps_id = (ps_id % len(data_files))
      data_files = data_files[ps_id:] + data_files[:ps_id]
      src_files = meta_files + data_files
    else:
      src_files = meta_files
    for src_path in src_files:
      _, file_name = os.path.split(src_path)
      dst_path = os.path.join(tmpdir, os.path.basename(src_path))
      logging.info('will copy %s to local path %s' % (src_path, dst_path))
      try:
        output = subprocess.check_output(
            'hadoop fs -get %s %s' % (src_path, dst_path), shell=True)
        logging.info('copy succeed: %s' % output)
      except Exception:
        logging.warning('exception: %s' % traceback.format_exc())
    ckpt_filename = os.path.basename(fine_tune_ckpt_path)
    fine_tune_ckpt_path = os.path.join(tmpdir, ckpt_filename)
    pipeline_config.train_config.fine_tune_checkpoint = fine_tune_ckpt_path
    logging.info('will restore from %s' % fine_tune_ckpt_path)
  else:
    # workers do not have to create the restore graph
    pipeline_config.train_config.ClearField('fine_tune_checkpoint')
