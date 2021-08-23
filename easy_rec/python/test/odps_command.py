# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import subprocess

from easy_rec.python.test.odps_test_util import get_oss_bucket


class OdpsCommand:

  def __init__(self, odps_oss_config):
    """Wrapper for running odps command.

    Args:
      odps_oss_config: instance of easy_rec.python.utils.odps_test_util.OdpsOSSConfig
    """
    self.bucket = get_oss_bucket(odps_oss_config.oss_key,
                                 odps_oss_config.oss_secret,
                                 odps_oss_config.endpoint,
                                 odps_oss_config.bucket_name)
    self.bucket_name = odps_oss_config.bucket_name
    self.temp_dir = odps_oss_config.temp_dir
    self.log_path = odps_oss_config.log_dir
    self.odpscmd = odps_oss_config.odpscmd_path
    self.odps_config_path = odps_oss_config.odps_config_path
    self.algo_project = odps_oss_config.algo_project
    self.algo_res_project = odps_oss_config.algo_res_project
    self.algo_version = odps_oss_config.algo_version

  def run_odps_cmd(self, script_file):
    """Run sql use odpscmd.

    Args:
      script_file: xxx.sql file, to be runned by odpscmd
    Raise:
      ValueError if failed
    """
    exec_file_path = os.path.join(self.temp_dir, script_file)
    file_name = os.path.split(script_file)[1]
    log_file = os.path.join(self.log_path, file_name)

    if self.odps_config_path is None:
      cmd = 'nohup %s  -f  %s > %s.log 2>&1' % (self.odpscmd, exec_file_path,
                                                log_file)
    else:
      cmd = 'nohup %s --config=%s -f  %s > %s.log 2>&1' % (
          self.odpscmd, self.odps_config_path, exec_file_path, log_file)
    logging.info('will run cmd: %s' % (cmd))
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    if (proc.returncode == 0):
      logging.info('%s run succeed' % script_file)
    else:
      raise ValueError('%s run FAILED: please check log file:%s.log' %
                       (exec_file_path, log_file))

  def run_list(self, files):
    for f in files:
      self.run_odps_cmd(f)
