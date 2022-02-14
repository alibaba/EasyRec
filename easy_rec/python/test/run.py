#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

tf.app.flags.DEFINE_bool('list_tests', False, 'list all tests')
tf.app.flags.DEFINE_string('list_test_to_file', None, 'list all tests')
tf.app.flags.DEFINE_string('pattern', '*_test.py', 'test file pattern')
tf.app.flags.DEFINE_string('test_dir', 'easy_rec/python/test',
                           'directory to be tested')
FLAGS = tf.flags.FLAGS


def gather_test_cases(test_dir, pattern):
  test_suite = unittest.TestSuite()
  discover = unittest.defaultTestLoader.discover(
      test_dir, pattern=pattern, top_level_dir=None)
  all_tests = []
  for suite_discovered in discover:

    for test_case in suite_discovered:
      test_suite.addTest(test_case)
      if hasattr(test_case, '__iter__'):
        for subcase in test_case:
          if FLAGS.list_tests or FLAGS.list_test_to_file:
            print(subcase.id())
            tid = subcase.id().split('.')[0]
            if tid not in all_tests:
              all_tests.append(tid)
      else:
        if FLAGS.list_tests or FLAGS.list_test_to_file:
          print(test_case.id())
          tid = subcase.id().split('.')[0]
          if tid not in all_tests:
            all_tests.append(tid)
  if FLAGS.list_test_to_file:
    print('save test lists to %s' % FLAGS.list_test_to_file)
    with open(FLAGS.list_test_to_file, 'w') as fout:
      for t_name in all_tests:
        fout.write('%s\n' % t_name)
  return test_suite


def main(argv):
  runner = unittest.TextTestRunner()
  test_suite = gather_test_cases(os.path.abspath(FLAGS.test_dir), FLAGS.pattern)
  if FLAGS.list_tests or FLAGS.list_test_to_file:
    return

  result = runner.run(test_suite)
  if not result.wasSuccessful():
    print('FailNum: %d ErrorNum: %d' %
          (len(result.failures), len(result.errors)))
  else:
    if 'UnitTestSucceedFlag' in os.environ:
      flag_file = os.environ['UnitTestSucceedFlag']
      with open(flag_file, 'w') as fout:
        fout.write('unit succeed.')
      print('create flag file: %s' % flag_file)


if __name__ == '__main__':
  tf.app.run()
