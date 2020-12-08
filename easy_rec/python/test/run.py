#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
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
    runner.run(test_suite)


if __name__ == '__main__':
  tf.app.run()
