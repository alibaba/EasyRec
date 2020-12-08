# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from multiprocessing import Process

from easy_rec.python.test.odps_command import OdpsCommand


class OdpsTest(Process):
  """Test easyrec on odps."""

  def __init__(self, start_files, test_files, end_file, odps_oss_config):
    """Init odps test instance.

    Args:
      start_files: prepare sql files, to create tables
      test_files: actual test sql files, such as train/eval/export
      end_file: drop table sql files
      odps_oss_config: easy_rec.python.test.odps_test_util.OdpsOSSConfig
    """
    super(OdpsTest, self).__init__()
    self.start_files = start_files
    self.test_files = test_files
    self.end_file = end_file

    self.odps = OdpsCommand(odps_oss_config)
    self.init_table()

  def init_table(self):
    """Init table for test."""
    self.odps.run_list(self.start_files)

  def start_test(self):
    """Exec every file in self.test_files."""
    self.odps.run_list(self.test_files)

  def drop_table(self):
    """Drop temporary tables."""
    self.odps.run_list(self.end_file)
