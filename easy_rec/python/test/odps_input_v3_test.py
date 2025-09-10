# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Test odps input v3."""

import tensorflow as tf
from easy_rec.python.input.odps_input_v3 import OdpsInputV3
from google.protobuf import text_format
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig


class OdpsInputV3Test(tf.test.TestCase):

  def __init__(self, methodName='OdpsInputV3'):
    super(OdpsInputV3Test, self).__init__(methodName=methodName)
    # must put .odps_config.ini in HOME directory
    self._input_path = 'odps://xingqudao_saas/tables/xingqudao_dbmtl_v1_training_set_fg_encoded_v1/ds=20250526'

  def test_read_data(self):
    data_config_str = """
         input_fields {
           input_name: 'user_id'
           input_type: STRING
         }
         input_fields {
           input_name: 'item_id'
           input_type: STRING
         }
         input_fields {
           input_name: 'is_click'
           input_type: INT64
         }
         input_fields {
           input_name: 'is_collect_like_comment'
           input_type: INT64
         }
         input_fields {
           input_name: 'features'
           input_type: STRING
         }
         label_fields: 'is_click'
         batch_size: 1024
         num_epochs: 1
         prefetch_size: 1
       """
    feature_config_str = """
         input_names: 'user_id'
         feature_type: IdFeature
         embedding_dim: 32
         hash_bucket_size: 2000
       """
    dataset_config = DatasetConfig()
    text_format.Merge(data_config_str, dataset_config)
    feature_config = FeatureConfig()
    text_format.Merge(feature_config_str, feature_config)
    feature_configs = [feature_config]
    empty_config = FeatureConfig()
    empty_config.CopyFrom(feature_config)
    while len(empty_config.input_names) > 0:
        empty_config.input_names.pop()
    for input_name in ['item_id', 'features']:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(input_name)
        feature_configs.append(tmp_config)

    train_input_fn = OdpsInputV3(dataset_config, feature_configs,
                                 self._input_path).create_input()
    dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()
    init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)
    with self.test_session(config=session_config) as sess:
      sess.run(init_op)
      feature_dict, label_dict = sess.run([features, labels])
      print("feature:", feature_dict, "label:", label_dict)


if __name__ == '__main__':
  tf.test.main()
