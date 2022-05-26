# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import os
import time

import numpy as np
import tensorflow as tf

from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils import hpo_util
from easy_rec.python.utils import test_utils

if tf.__version__ >= '2.0':
  gfile = tf.compat.v1.gfile
  from tensorflow.core.protobuf import config_pb2

  ConfigProto = config_pb2.ConfigProto
  GPUOptions = config_pb2.GPUOptions
else:
  gfile = tf.gfile
  GPUOptions = tf.GPUOptions
  ConfigProto = tf.ConfigProto


class HPOTest(tf.test.TestCase):

  def __init__(self, methodName='HPOTest'):
    super(HPOTest, self).__init__(methodName=methodName)
    self._metric_data_path = 'data/test/hpo_test/eval_val/*.tfevents.*'

  def test_get_metric(self):
    vals = hpo_util.get_all_eval_result(self._metric_data_path)
    logging.info('eval result num = %d' % len(vals))
    logging.info('eval result[0] = %s' % json.dumps(vals[0]))

  def load_config(self, config_path):
    with gfile.GFile(config_path, 'r') as fin:
      return json.load(fin)['param']

  def test_save_eval_metrics(self):
    test_dir = test_utils.get_tmp_dir()
    tmp_file = os.path.join(test_dir,
                            'easy_rec_hpo_test_%d.metric' % time.time())
    hpo_util.save_eval_metrics('data/test/hpo_test/', tmp_file, False)
    test_utils.clean_up(test_dir)

  def test_edit_config(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    assert tmp_config.feature_config.features[0].embedding_dim == 120

  def test_edit_config_v2(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v2.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for tmp_fea in tmp_config.feature_configs:
      if tmp_fea.input_names[0] == 'site_id':
        assert tmp_fea.embedding_dim == 32
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v3(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v3.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i >= 10 and i < 20:
        assert tmp_fea.embedding_dim == 37
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v4(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v4.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i < 15:
        assert tmp_fea.embedding_dim == 37
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v5(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v5.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i >= 5:
        assert tmp_fea.embedding_dim == 37
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v51(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v51.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i == 5:
        assert tmp_fea.embedding_dim == 37

  def test_edit_config_v6(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v6.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] >= 'site':
        assert tmp_fea.embedding_dim == 32, 'input_name = %s %d' % (
            tmp_fea.input_names[0], tmp_fea.embedding_dim)
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v7(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v7.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'c21':
        assert len(tmp_fea.boundaries) == 4 and np.abs(tmp_fea.boundaries[0] -
                                                       10.0) < 1e-5

  def test_edit_config_v71(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v71.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'c21':
        assert len(tmp_fea.boundaries) == 4 and np.abs(tmp_fea.boundaries[0] -
                                                       10.0) < 1e-5

  def test_edit_config_v8(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v8.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'c21':
        assert len(tmp_fea.boundaries) == 4 and np.abs(tmp_fea.boundaries[0] -
                                                       4.0) < 1e-5
        assert tmp_fea.embedding_dim == 32

  def test_edit_config_v81(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v81.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.feature_type == tmp_fea.RawFeature:
        assert tmp_fea.embedding_dim == 24

  def test_edit_config_v9(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v9.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    assert tmp_config.train_config.fine_tune_checkpoint == \
           'oss://easy-rec/test/experiment/ctr_v93/model.ckpt-1000'

  def test_edit_config_v10(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v10.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'c21':
        assert len(tmp_fea.boundaries) == 4 and np.abs(tmp_fea.boundaries[0] -
                                                       4.0) < 1e-5
        assert tmp_fea.embedding_dim == 32

  def test_edit_config_v11(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v11.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'c21':
        assert len(tmp_fea.boundaries) == 4 and np.abs(tmp_fea.boundaries[0] -
                                                       10.0) < 1e-5

  def test_edit_config_v12(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v12.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'c21':
        assert len(tmp_fea.boundaries) == 25
        assert np.abs(tmp_fea.boundaries[1] - 21.0) < 1e-5

  def test_edit_config_v13(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v13.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    assert not tmp_config.export_config.multi_placeholder

  def test_edit_config_v14(self):
    tmp_file = 'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v14.json'
    tmp_config = config_util.edit_config(tmp_config, self.load_config(tmp_file))
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] == 'hour':
        assert len(tmp_fea.feature_type) == FeatureConfig.RawFeature

  def test_save_eval_metrics_with_env(self):
    os.environ['TF_CONFIG'] = """
                              { "cluster": {
                                  "worker": ["127.0.0.1:2020"],
                                  "chief": ["127.0.0.1:2021"]
                                 },
                                 "task": {"type": "chief", "index": 0}
                              }
                              """
    test_dir = test_utils.get_tmp_dir()
    tmp_file = os.path.join(test_dir,
                            'easy_rec_hpo_test_%d.metric' % time.time())
    hpo_util.save_eval_metrics('data/test/hpo_test/', tmp_file, False)
    test_utils.clean_up(test_dir)


if __name__ == '__main__':
  tf.test.main()
