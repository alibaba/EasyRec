# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.input.dummy_input import DummyInput
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class EmbedTest(tf.test.TestCase):

  def test_raw_embed(self):
    # embedding variable is:
    #    [[1, 2 ],
    #     [3, 4 ],
    #     [5, 6 ],
    #     [7, 8 ],
    #     [9, 10]
    #    ]
    feature_config_str = '''
      input_names: 'field1'
      feature_type: RawFeature
      initializer {
         constant_initializer {
            consts: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
         }
      }
      separator: ',',
      raw_input_dim: 5
      embedding_dim: 2
      combiner: 'sum'
    '''
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)

    data_config_str = '''
        input_fields {
           input_name: 'clk'
           input_type: INT32
           default_val: '0'
        }
        input_fields {
           input_name: 'field1'
           input_type: STRING
           default_val: '0'
        }
        label_fields: 'clk'
        batch_size: 1
    '''
    data_config = DatasetConfig()
    text_format.Merge(data_config_str, data_config)

    feature_configs = [feature_config]
    features = {'field1': tf.constant(['0.1,0.2,0.3,0.4,0.5'])}
    dummy_input = DummyInput(
        data_config, feature_configs, '', input_vals=features)
    field_dict, _ = dummy_input._build(tf.estimator.ModeKeys.TRAIN, {})

    wide_and_deep_dict = {'field1': WideOrDeep.WIDE_AND_DEEP}
    fc_parser = FeatureColumnParser(feature_configs, wide_and_deep_dict, 2)
    wide_cols = list(fc_parser._wide_columns.values())
    wide_features = feature_column.input_layer(field_dict, wide_cols)
    deep_cols = list(fc_parser._deep_columns.values())
    deep_features = feature_column.input_layer(field_dict, deep_cols)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init)
      fea_val = sess.run(wide_features)
      logging.info('wide fea_val = %s' % str(fea_val[0]))
      assert np.abs(fea_val[0][0] - 9.5) < 1e-6
      assert np.abs(fea_val[0][1] - 11.0) < 1e-6
      fea_val = sess.run(deep_features)
      logging.info('deep fea_val = %s' % str(fea_val[0]))
      assert np.abs(fea_val[0][0] - 9.5) < 1e-6
      assert np.abs(fea_val[0][1] - 11.0) < 1e-6

  def test_seq_multi_embed(self):
    # embedding variable is:
    #    [[1, 2 ],
    #     [3, 4 ],
    #     [5, 6 ],
    #     [7, 8 ],
    #     [9, 10]
    #    ]
    feature_config_str = '''
      input_names: 'field1'
      feature_type: SequenceFeature
      initializer {
         constant_initializer {
            consts: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
         }
      }
      separator: '',
      seq_multi_sep: '',
      embedding_dim: 2
      num_buckets: 5
      combiner: 'mean'
    '''
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)

    data_config_str = '''
        input_fields {
           input_name: 'clk'
           input_type: INT32
           default_val: '0'
        }
        input_fields {
           input_name: 'field1'
           input_type: STRING
           default_val: '0'
        }
        label_fields: 'clk'
        batch_size: 1
    '''
    data_config = DatasetConfig()
    text_format.Merge(data_config_str, data_config)

    feature_configs = [feature_config]
    features = {'field1': tf.constant(['0112', '132430'])}
    dummy_input = DummyInput(
        data_config, feature_configs, '', input_vals=features)
    field_dict, _ = dummy_input._build(tf.estimator.ModeKeys.TRAIN, {})

    wide_and_deep_dict = {'field1': WideOrDeep.DEEP}
    fc_parser = FeatureColumnParser(feature_configs, wide_and_deep_dict)
    builder = feature_column._LazyBuilder(field_dict)
    hist_embedding, hist_seq_len = \
        fc_parser.sequence_columns['field1']._get_sequence_dense_tensor(builder)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init)
      fea_val, len_val = sess.run([hist_embedding, hist_seq_len])
      logging.info('length_val = %s' % str(len_val))
      logging.info('deep fea_val = %s' % str(fea_val))
      assert np.abs(fea_val[0][0][0] - 2) < 1e-6
      assert np.abs(fea_val[0][0][1] - 3) < 1e-6
      assert np.abs(fea_val[0][1][0] - 4) < 1e-6
      assert np.abs(fea_val[0][1][1] - 5) < 1e-6


if __name__ == '__main__':
  tf.test.main()
