# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Define cv_input, the base class for cv tasks."""
import logging
import os
import unittest

import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.input.hive_input import HiveInput
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.protos.hive_config_pb2 import HiveConfig
from easy_rec.python.protos.pipeline_pb2 import EasyRecConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils import test_utils

if tf.__version__ >= '2.0':
  import tensorflow.compat.v1 as tf

gfile = tf.gfile

if tf.__version__ >= '2.0':
  from tensorflow.python.framework.ops import disable_eager_execution

  disable_eager_execution()
  tf = tf.compat.v1


class HiveInputTest(tf.test.TestCase):

  def _init_config(self):
    hive_host = os.environ['hive_host']
    hive_username = os.environ['hive_username']
    hive_table_name = os.environ['hive_table_name']
    hive_hash_fields = os.environ['hive_hash_fields']

    hive_train_input = """
      host: "{}"
      username: "{}"
      table_name: "{}"
      limit_num: 500
      hash_fields: "{}"
    """.format(hive_host, hive_username, hive_table_name, hive_hash_fields)
    hive_eval_input = """
      host: "{}"
      username: "{}"
      table_name: "{}"
      limit_num: 500
      hash_fields: "{}"
    """.format(hive_host, hive_username, hive_table_name, hive_hash_fields)
    self.hive_train_input_config = HiveConfig()
    text_format.Merge(hive_train_input, self.hive_train_input_config)

    self.hive_eval_input_config = HiveConfig()
    text_format.Merge(hive_eval_input, self.hive_eval_input_config)

  def __init__(self, methodName='HiveInputTest'):
    super(HiveInputTest, self).__init__(methodName=methodName)

  @unittest.skipIf('hive_host' not in os.environ or
                   'hive_username' not in os.environ or
                   'hive_table_name' not in os.environ or
                   'hive_hash_fields' not in os.environ,
                   """Only execute hive_config var are specified,hive_host、
       hive_username、hive_table_name、hive_hash_fields is available.""")
  def test_hive_input(self):
    self._init_config()
    data_config_str = """
          batch_size: 1024
          label_fields: "label_1"
          label_fields: "label_2"
          num_epochs: 1
          prefetch_size: 32
          input_type: HiveInput
          input_fields {
            input_name:'label_1'
            input_type: INT32
          }
          input_fields {
            input_name:'label_2'
            input_type: INT32
          }
          input_fields {
            input_name:'age'
            input_type: INT32
          }
          input_fields {
            input_name: "class_of_worker"
          }
          input_fields {
            input_name: "industry_code"
          }
          input_fields {
            input_name: "occupation_code"
          }
          input_fields {
            input_name: "education"
          }
          input_fields {
            input_name: "wage_per_hour"
            input_type: DOUBLE
          }
          input_fields {
            input_name: "enrolled_in_edu_inst_last_wk"
          }
          input_fields {
            input_name: "major_industry"
          }
          input_fields {
            input_name: "major_occupation"
          }
          input_fields {
            input_name: "mace"
          }
          input_fields {
            input_name: "hispanic_origin"
          }
          input_fields {
            input_name: "sex"
          }
          input_fields {
            input_name: "member_of_a_labor_union"
          }
          input_fields {
            input_name: "reason_for_unemployment"
          }
          input_fields {
            input_name: "full_or_part_time_employment_stat"
          }
          input_fields {
            input_name: "capital_gains"
            input_type: DOUBLE
          }
          input_fields {
            input_name: "capital_losses"
            input_type: DOUBLE
          }
          input_fields {
            input_name: "divdends_from_stocks"
            input_type: DOUBLE
          }
          input_fields {
            input_name: "tax_filer_status"
          }
          input_fields {
            input_name: "region_of_previous_residence"
          }
          input_fields {
            input_name: "state_of_previous_residence"
          }
          input_fields {
            input_name: "detailed_household_and_family_stat"
          }
          input_fields {
            input_name: "detailed_household_summary_in_household"
          }
          input_fields {
            input_name: "instance_weight"
          }
          input_fields {
            input_name: "migration_code_change_in_msa"
          }
          input_fields {
            input_name: "migration_code_change_in_reg"
          }
          input_fields {
            input_name: "migration_code_move_within_reg"
          }
          input_fields {
            input_name: "live_in_this_house_1_year_ago"
          }
          input_fields {
            input_name: "migration_prev_res_in_sunbelt"
          }
          input_fields {
            input_name: "num_persons_worked_for_employer"
            input_type: INT32
          }
          input_fields {
            input_name: "family_members_under_18"
          }
          input_fields {
            input_name: "country_of_birth_father"
          }
          input_fields {
            input_name: "country_of_birth_mother"
          }
          input_fields {
            input_name: "country_of_birth_self"
          }
          input_fields {
            input_name: "citizenship"
          }
          input_fields {
            input_name: "own_business_or_self_employed"
          }
          input_fields {
            input_name: "fill_inc_questionnaire_for_veteran_s_admin"
          }
          input_fields {
            input_name: "veterans_benefits"
          }
          input_fields {
            input_name: "weeks_worked_in_year"
            input_type: INT32
          }
          input_fields {
            input_name: "year"
          }
      """

    feature_config_str = """
      input_names: "own_business_or_self_employed"
      feature_type: IdFeature
      embedding_dim: 9
      hash_bucket_size: 400
      embedding_name: "feature"
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
    while len(empty_config.shared_names) > 0:
      empty_config.shared_names.pop()
    train_input_fn = HiveInput(dataset_config, feature_configs,
                               self.hive_train_input_config).create_input()
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
      for key in feature_dict:
        print(key, feature_dict[key][:5])

      for key in label_dict:
        print(key, label_dict[key][:5])
    return 0

  @unittest.skipIf('hive_host' not in os.environ or
                   'hive_username' not in os.environ or
                   'hive_table_name' not in os.environ or
                   'hive_hash_fields' not in os.environ,
                   """Only execute hive_config var are specified,hive_host、
       hive_username、hive_table_name、hive_hash_fields is available.""")
  def test_mmoe(self):
    pipeline_config_path = 'samples/emr_script/mmoe/mmoe_census_income.config'
    gpus = test_utils.get_available_gpus()
    if len(gpus) > 0:
      test_utils.set_gpu_id(gpus[0])
    else:
      test_utils.set_gpu_id(None)

    if not isinstance(pipeline_config_path, EasyRecConfig):
      logging.info('testing pipeline config %s' % pipeline_config_path)
    if 'TF_CONFIG' in os.environ:
      del os.environ['TF_CONFIG']

    if isinstance(pipeline_config_path, EasyRecConfig):
      pipeline_config = pipeline_config_path
    else:
      pipeline_config = test_utils._load_config_for_test(
          pipeline_config_path, self._test_dir)

    pipeline_config.train_config.train_distribute = 0
    pipeline_config.train_config.num_gpus_per_worker = 1
    pipeline_config.train_config.sync_replicas = False

    config_util.save_pipeline_config(pipeline_config, self._test_dir)
    test_pipeline_config_path = os.path.join(self._test_dir, 'pipeline.config')
    hyperparam_str = ''
    train_cmd = 'python -m easy_rec.python.train_eval --pipeline_config_path %s %s' % (
        test_pipeline_config_path, hyperparam_str)
    proc = test_utils.run_cmd(train_cmd,
                              '%s/log_%s.txt' % (self._test_dir, 'master'))
    proc.wait()
    if proc.returncode != 0:
      logging.error('train %s failed' % test_pipeline_config_path)
      return 1
    return 0

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    self._test_dir = test_utils.get_tmp_dir()
    self._success = True
    logging.info('test dir: %s' % self._test_dir)

  def tearDown(self):
    test_utils.set_gpu_id(None)
    if self._success:
      test_utils.clean_up(self._test_dir)


if __name__ == '__main__':
  tf.test.main()
