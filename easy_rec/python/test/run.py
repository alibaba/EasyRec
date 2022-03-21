#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import unittest

import tensorflow as tf

from easy_rec.python.utils import test_utils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

tf.app.flags.DEFINE_bool('list_tests', False, 'list all tests')
tf.app.flags.DEFINE_string('list_test_to_file', None, 'list all tests')
tf.app.flags.DEFINE_string('pattern', '*_test.py', 'test file pattern')
tf.app.flags.DEFINE_string('test_dir', 'easy_rec/python/test',
                           'directory to be tested')
tf.app.flags.DEFINE_integer('num_parallel', 10,
                            'number of parallel executed cases.')
tf.app.flags.DEFINE_integer('timeout', 3600,
                            'maximal execute time in seconds for each case.')
FLAGS = tf.flags.FLAGS


def gather_test_cases(test_dir, pattern):
  discover = unittest.defaultTestLoader.discover(
      test_dir, pattern=pattern, top_level_dir=None)
  all_tests = []
  for suite_discovered in discover:
    for test_case in suite_discovered:
      if hasattr(test_case, '__iter__'):
        for subcase in test_case:
          toks = subcase.id().split('.')
          case_file = toks[0]
          case_name = '.'.join(toks[1:])
          if (case_file, case_name) not in all_tests:
            all_tests.append((case_file, case_name))
      else:
        toks = subcase.id().split('.')[0]
        case_file = toks[0]
        case_name = '.'.join(toks[1:])
        if (case_file, case_name) not in all_tests:
          all_tests.append((case_file, case_name))
  if FLAGS.list_test_to_file:
    logging.info('Total number of cases: %d' % len(all_tests))
    logging.info('save test lists to %s' % FLAGS.list_test_to_file)
    with open(FLAGS.list_test_to_file, 'w') as fout:
      for t_file, t_name in all_tests:
        fout.write('%s %s\n' % (t_file, t_name))
  elif FLAGS.list_tests:
    logging.info('Total number of cases: %d' % len(all_tests))
    for t_file, t_name in all_tests:
      logging.info('\t%s.%s' % (t_file, t_name))
  return all_tests


def main(argv):
  all_tests = gather_test_cases(os.path.abspath(FLAGS.test_dir), FLAGS.pattern)
  if FLAGS.list_tests or FLAGS.list_test_to_file:
    return

  test_dir = os.environ.get('TEST_DIR', '.')
  if not os.path.isdir(test_dir):
    os.makedirs(test_dir)
  test_log_dir = os.path.join(test_dir, 'logs')
  os.makedirs(test_log_dir)
  logging.info('Total number of cases: %d test_dir: %s' %
               (len(all_tests), test_dir))
  procs = {}
  failed_cases = []
  for case_file, case_name in all_tests:
    while len(procs) >= FLAGS.num_parallel:
      procs_done = []
      for proc in procs:
        if proc.poll() is not None:
          if proc.returncode != 0:
            fail_file, fail_name = procs[proc]
            failed_cases.append((fail_file, fail_name, proc.returncode))
          procs_done.append(proc)
      for proc in procs_done:
        del procs[proc]
    cmd = 'python -m easy_rec.python.test.%s %s' % (case_file, case_name)
    log_file = '%s/%s.%s.log' % (test_log_dir, case_file, case_name)
    logging.info('Run %s.%s Log: %s' % (case_file, case_name, log_file))
    proc = test_utils.run_cmd(cmd, log_file)
    procs[proc] = (case_file, case_name)

  for proc in procs:
    try:
      proc.wait()
    except Exception as ex:
      fail_file, fail_name = procs[proc]
      logging.info('Case Exception: %s.%s %s' % (fail_file, fail_name, str(ex)))
      proc.kill()

    if proc.returncode != 0:
      fail_file, fail_name = procs[proc]
      failed_cases.append((fail_file, fail_name, proc.returncode))

  if len(failed_cases) > 0:
    logging.info('Number Cases Failed: %d' % len(failed_cases))
    for fail_file, fail_name, exit_code in failed_cases:
      logging.info('\t%s.%s failed, exit_code:%d log: %s.%s.log' %
                   (fail_file, fail_name, exit_code, fail_file, fail_name))
    return 1
  else:
    logging.info('TestSucceed.')
    return 0


if __name__ == '__main__':
  tf.app.run()
