# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Define cv_input, the base class for cv tasks."""

import tensorflow as tf

from google.protobuf import text_format
from easy_rec.python.input.hive_input import HiveInput
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils.test_utils import RunAsSubprocess


if tf.__version__ >= '2.0':
  from tensorflow.python.framework.ops import disable_eager_execution

  disable_eager_execution()
  tf = tf.compat.v1


class HiveInputTest(tf.test.TestCase):

  def __init__(self, methodName='HiveInputTest'):
    super(HiveInputTest, self).__init__(methodName=methodName)
    self._input_path = 'census_income_train_simple'

  @RunAsSubprocess
  def test_hive_input(self):
    data_config_str = """
        batch_size: 1024
        label_fields: "label_1"
        label_fields: "label_2"
        num_epochs: 1
        prefetch_size: 32
        input_type: HiveInput
        hive_config {
          host: "localhost"
          username: "zhenghong"
          hash_fields: "age,class_of_worker, marital_status,education"
        }
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
    for input_name in feature_config.shared_names:
      input_names = config_util.auto_expand_names(input_name)
      for tmp_name in input_names:
        tmp_config = FeatureConfig()
        tmp_config.CopyFrom(empty_config)
        tmp_config.input_names.append(tmp_name)
        feature_configs.append(tmp_config)
    train_input_fn = HiveInput(dataset_config, feature_configs,
                               self._input_path).create_input()
    dataset = train_input_fn(mode=tf.estimator.ModeKeys.TRAIN)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    # features, labels = iterator.get_next()
    # init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # session_config = tf.ConfigProto(
    #     gpu_options=gpu_options,
    #     allow_soft_placement=True,
    #     log_device_placement=False)
    # with self.test_session(config=session_config) as sess:
    #   sess.run(init_op)
    #   feature_dict, label_dict = sess.run([features, labels])
    #   for key in feature_dict:
    #     print(key, feature_dict[key][:5])
    #
    #   for key in label_dict:
    #     print(key, label_dict[key][:5])


if __name__ == '__main__':
  tf.test.main()
