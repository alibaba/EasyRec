# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import division

import logging
import unittest

from google.protobuf import text_format

from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.easy_rec_model_pb2 import EasyRecModel as EasyRecModelProto
from easy_rec.python.protos.summary_pb2 import ModelSummaries

if __name__ == '__main__':
  import tensorflow as tf
  if tf.__version__ >= '2.0':
    tf = tf.compat.v1
  tf.disable_eager_execution()


def _tf_v1():
  import tensorflow as tf
  return tf.compat.v1 if tf.__version__ >= '2.0' else tf


class _FakeEasyRecModel(object):
  _resolve_pred_tensor = EasyRecModel._resolve_pred_tensor
  _build_feature_mask = EasyRecModel._build_feature_mask
  _masked_mean = EasyRecModel._masked_mean
  _get_summary_label_name = EasyRecModel._get_summary_label_name


def _run_summary_tags(model, summary_text):
  tf = _tf_v1()
  tf.reset_default_graph()
  model = model()
  summary = ModelSummaries()
  text_format.Parse(summary_text, summary)
  EasyRecModel._build_summary_impl(model, summary)
  with tf.Session() as sess:
    summary_str = sess.run(tf.summary.merge_all())
  summary_proto = tf.Summary()
  summary_proto.ParseFromString(summary_str)
  return {v.tag: v.simple_value for v in summary_proto.value}


class SummariesSetTest(unittest.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def test_proto_and_pcoc_graph(self):
    model = EasyRecModelProto()
    text_format.Parse(
        'model_class: "DeepFM" deepfm {} summaries_set { pcoc {} }', model)
    self.assertEqual(model.summaries_set[0].WhichOneof('summary'), 'pcoc')

    class _M(_FakeEasyRecModel):

      def __init__(self):
        tf = _tf_v1()
        self._labels = {'label': tf.constant([1, 0, 1, 0], dtype=tf.float32)}
        self._prediction_dict = {
            'probs': tf.constant([0.8, 0.4, 0.6, 0.2], dtype=tf.float32)
        }
        self._feature_dict = {}
        self._label_name = 'label'
        self._base_model_config = EasyRecModelProto()

    values = _run_summary_tags(_M, 'pcoc { epsilon: 1e-7 }')
    self.assertAlmostEqual(values['summary/pcoc'], 1.0, places=5)

  def test_scalars_feature_slice(self):
    class _M(_FakeEasyRecModel):

      def __init__(self):
        tf = _tf_v1()
        self._labels = {'label': tf.constant([1, 0], dtype=tf.float32)}
        self._prediction_dict = {
            'probs': tf.constant([0.9, 0.1], dtype=tf.float32)
        }
        self._feature_dict = {'c1': tf.constant([1005.0, 1002.0], dtype=tf.float32)}
        self._label_name = 'label'
        self._base_model_config = EasyRecModelProto()

    values = _run_summary_tags(
        _M,
        'scalars { name: "c1_1005_pred" feature_name: "c1" feature_value: "1005" }')
    self.assertAlmostEqual(values['summary/c1_1005_pred'], 0.9, places=5)

  def test_pcoc_without_rank_label_name(self):
    """MatchModel 等无 _label_name 时，回退到 model_config.label_name。"""

    class _M(_FakeEasyRecModel):

      def __init__(self):
        tf = _tf_v1()
        self._labels = {'clk': tf.constant([1, 0], dtype=tf.float32)}
        self._prediction_dict = {
            'probs': tf.constant([0.6, 0.4], dtype=tf.float32)
        }
        self._feature_dict = {}
        self._base_model_config = EasyRecModelProto()
        self._base_model_config.label_name = 'clk'

    values = _run_summary_tags(_M, 'pcoc {}')
    self.assertAlmostEqual(values['summary/pcoc'], 1.0, places=5)

  def test_pcoc_custom_pred_name(self):
    class _M(_FakeEasyRecModel):

      def __init__(self):
        tf = _tf_v1()
        self._labels = {'label': tf.constant([1, 0], dtype=tf.float32)}
        self._prediction_dict = {
            'similarity': tf.constant([0.75, 0.25], dtype=tf.float32)
        }
        self._feature_dict = {}
        self._label_name = 'label'
        self._base_model_config = EasyRecModelProto()

    values = _run_summary_tags(_M, 'pcoc { pred_name: "similarity" }')
    self.assertAlmostEqual(values['summary/pcoc'], 1.0, places=5)


if __name__ == '__main__':
  unittest.main()
