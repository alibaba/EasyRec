# -*- encoding: utf-8 -*-
import logging
import os
import signal
import threading
import time


class OdlWatchDog:

  def __init__(self, max_idle_period):
    self._last_update_time = 0
    self._run = False

    def _check_func():
      logging.info('OdlWatchDog started')
      while self._run:
        ts = time.time()
        if self._last_update_time > 0 and ts - self._last_update_time > max_idle_period:
          logging.error(
              'max_idle_period reached: current time[%d] last_update_time[%d]'
              ' max_idle_period=%d' %
              (int(ts), int(self._last_update_time), max_idle_period))
          os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(60)
      logging.info('OdlWatchDog will exit')

    self._check_thread = threading.Thread(target=_check_func)

  def start(self):
    logging.info('start OdlWatchDog')
    self._run = True
    self._check_thread.start()

  def stop(self):
    self._run = False
    self._check_thread.join()

  def set_last_update(self):
    self._last_update_time = time.time()


class ErrorLogWatchDog:

  def __init__(self, log_path='../stderr.out'):
    self._run = False
    self._stderr_path = log_path
    logging.info('log_path: %s exists=%d' %
                 (self._stderr_path, os.path.exists(self._stderr_path)))
    self._error_pattern = 'This error may also occur due to a gRPC failure caused by high memory or network bandwidth usage in the parameter servers.'  # noqa: E501
    self._error_pattern_1 = 'tensorflow.python.framework.errors_impl.InternalError: Missing 0-th output from node GetSparseIndices'  # noqa: E501
    # check last 10k
    self._max_chk_size = 10240

    def _check_func():
      logging.info('ErrorLogWatchDog started')
      while self._run:
        if self.has_error_log():
          os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(1200)
      logging.info('ErrorLogWatchDog will exit')

    self._check_thread = threading.Thread(target=_check_func)

  def has_error_log(self):
    if os.path.exists(self._stderr_path):
      with open(self._stderr_path, 'r') as fin:
        fin.seek(0, os.SEEK_END)
        file_size = fin.tell()
        seek_pos = max(file_size - self._max_chk_size, 0)
        logging.info('log_file_size=%d seek_pos=%d' % (file_size, seek_pos))
        fin.seek(seek_pos)
        for line_str in fin:
          if self._error_pattern in line_str:
            logging.error('error_pattern find: %s' % line_str.strip())
            return True
          elif self._error_pattern_1 in line_str:
            logging.error('error_pattern find: %s' % line_str.strip())
            return True
    return False

  def start(self):
    logging.info('start ErrorLogWatchDog')
    self._run = True
    self._check_thread.start()

  def stop(self):
    self._run = False
    self._check_thread.join()
