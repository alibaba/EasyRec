#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
import unittest

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

tf.app.flags.DEFINE_bool('list_tests', False, 'list all tests')
tf.app.flags.DEFINE_string('pattern', '*_test.py', 'test file pattern')
tf.app.flags.DEFINE_string('test_dir', 'easy_rec/python/test',
                           'directory to be tested')
FLAGS = tf.flags.FLAGS


def gather_test_cases(test_dir, pattern):
  test_suite = unittest.TestSuite()
  discover = unittest.defaultTestLoader.discover(
      test_dir, pattern=pattern, top_level_dir=None)
  for suite_discovered in discover:

    for test_case in suite_discovered:
      if 'train_eval_test.TrainEvalTest' in str(test_case):
        continue
      if 'predictor_test.' in str(test_case):
        continue
      test_suite.addTest(test_case)
      if hasattr(test_case, '__iter__'):
        for subcase in test_case:
          if FLAGS.list_tests:
            print(subcase)
      else:
        if FLAGS.list_tests:
          print(test_case)
  return test_suite


def main(argv):
  runner = unittest.TextTestRunner()
  test_suite = gather_test_cases(os.path.abspath(FLAGS.test_dir), FLAGS.pattern)
  if not FLAGS.list_tests:
    result = runner.run(test_suite)
    if not result.wasSuccessful():
      print('FailNum: %d ErrorNum: %d' % (len(result.failures), len(result.errors)))
    else:
      if 'UnitTestSucceedFlag' in os.environ:
        flag_file = os.environ['UnitTestSucceedFlag']
        with open(flag_file, 'w') as fout:
          fout.write('unit succeed.')
        print('create flag file: %s' % flag_file)
      

if __name__ == '__main__':
  tf.app.run()
