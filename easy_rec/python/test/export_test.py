# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Date: 2020-10-06
# Filename：export_test.py
import functools
import json
import logging
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import easy_rec
from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.utils import config_util
from easy_rec.python.utils import test_utils
from easy_rec.python.utils.test_utils import RunAsSubprocess


class ExportTest(tf.test.TestCase):

  def setUp(self):
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def tearDown(self):
    test_utils.set_gpu_id(None)

  @RunAsSubprocess
  def _predict_and_check(self,
                         data_path,
                         saved_model_dir,
                         cmp_result,
                         keys=['probs'],
                         separator=',',
                         tol=1e-4):
    predictor = Predictor(saved_model_dir)
    with open(data_path, 'r') as fin:
      inputs = []
      for line_str in fin:
        line_str = line_str.strip()
        if len(predictor.input_names) > 1:
          inputs.append(line_str.split(separator))
        else:
          inputs.append(line_str)
      output_res = predictor.predict(inputs, batch_size=32)

    for i in range(len(output_res)):
      for key in keys:
        val0 = output_res[i][key]
        val1 = cmp_result[i][key]
        diff = np.max(np.abs(val0 - val1))
        assert diff < tol, \
            'too much difference: %.6f for %s, tol=%.6f' \
            % (diff, key, tol)

  def _extract_data(self, input_path, output_path, offset=1, separator=','):
    with open(input_path, 'r') as fin:
      with open(output_path, 'w') as fout:
        for line_str in fin:
          line_str = line_str.strip()
          line_toks = line_str.split(separator)
          if offset > 0:
            line_toks = line_toks[offset:]
          fout.write('%s\n' % (separator.join(line_toks)))

  def _extract_rtp_data(self, input_path, output_path, separator=';'):
    with open(input_path, 'r') as fin:
      with open(output_path, 'w') as fout:
        for line_str in fin:
          line_str = line_str.strip()
          line_toks = line_str.split(separator)
          fout.write('%s\n' % line_toks[-1])

  def test_multi_tower(self):
    self._export_test('samples/model_config/multi_tower_export.config',
                      self._extract_data)

  def test_filter_input(self):
    self._export_test('samples/model_config/export_filter_input.config',
                      self._extract_data)

  def test_mmoe(self):
    self._export_test(
        'samples/model_config/mmoe_on_taobao.config',
        functools.partial(self._extract_data, offset=2),
        keys=['probs_ctr', 'probs_cvr'])

  def test_fg(self):
    self._export_test(
        'samples/model_config/taobao_fg.config',
        self._extract_rtp_data,
        separator='')

  def test_fg_export(self):
    self._export_test(
        'samples/model_config/taobao_fg_export.config',
        self._extract_rtp_data,
        separator='',
        test_multi=False)

  def test_export_with_asset(self):
    pipeline_config_path = 'samples/model_config/taobao_fg.config'
    test_dir = test_utils.get_tmp_dir()
    # prepare model
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.set_gpu_id(None)
    config_path = os.path.join(test_dir, 'pipeline.config')
    export_dir = os.path.join(test_dir, 'export/')
    export_cmd = """
      python -m easy_rec.python.export
        --pipeline_config_path %s
        --export_dir %s
        --asset_files fg.json:samples/model_config/taobao_fg.json
        --export_done_file ExportDone
    """ % (
        config_path,
        export_dir,
    )
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_%s.txt' % (test_dir, 'export'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    files = gfile.Glob(export_dir + '*')
    export_dir = files[0]
    assert gfile.Exists(export_dir + '/assets/taobao_fg.json')
    assert gfile.Exists(export_dir + '/assets/pipeline.config')
    assert gfile.Exists(export_dir + '/ExportDone')

  def test_export_with_out_in_ckpt_config(self):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)

    pipeline_config_path = 'samples/model_config/mmoe_on_taobao.config'

    def _post_check_func(pipeline_config):
      ckpt_path = tf.train.latest_checkpoint(pipeline_config.model_dir)
      export_dir = os.path.join(test_dir, 'train/export/no_config')
      export_cmd = """
        python -m easy_rec.python.export
          --pipeline_config_path %s
          --checkpoint_path %s
          --export_dir %s
      """ % (pipeline_config_path, ckpt_path, export_dir)
      proc = test_utils.run_cmd(export_cmd,
                                '%s/log_%s.txt' % (test_dir, 'export'))
      proc.wait()
      return proc.returncode == 0

    # prepare model
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path,
            test_dir=test_dir,
            post_check_func=_post_check_func))

  def test_multi_class_predict(self):
    self._export_test(
        'samples/model_config/deepfm_multi_cls_on_avazu_ctr.config',
        extract_data_func=self._extract_data,
        keys=['probs', 'logits', 'probs_y', 'logits_y', 'y'])

  def _export_test(self,
                   pipeline_config_path,
                   extract_data_func=None,
                   separator=',',
                   keys=['probs'],
                   test_multi=True):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)

    # prepare model
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir))
    test_utils.set_gpu_id(None)

    # prepare two version config
    config_path_single = os.path.join(test_dir, 'pipeline.config')
    config_path_multi = os.path.join(test_dir, 'pipeline_v2.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        config_path_single)
    if pipeline_config.export_config.multi_placeholder:
      config_path_single, config_path_multi = config_path_multi, config_path_single
    pipeline_config.export_config.multi_placeholder =\
        not pipeline_config.export_config.multi_placeholder
    config_util.save_pipeline_config(pipeline_config, test_dir,
                                     'pipeline_v2.config')

    # prepare two version export dir
    export_dir_single = os.path.join(test_dir, 'train/export/final')
    export_dir_multi = os.path.join(test_dir, 'train/export/multi')
    export_cmd = """
      python -m easy_rec.python.export
        --pipeline_config_path %s
        --export_dir %s
    """ % (config_path_multi, export_dir_multi)
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_%s.txt' % (test_dir, 'export'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    # use checkpoint to prepare result
    result_path = os.path.join(test_dir, 'result.txt')
    predict_cmd = """
      python -m easy_rec.python.predict
        --pipeline_config_path %s
        --output_path %s
    """ % (config_path_single, result_path)
    proc = test_utils.run_cmd(predict_cmd % (),
                              '%s/log_%s.txt' % (test_dir, 'predict'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    with open(result_path, 'r') as fin:
      cmp_result = []
      for line_str in fin:
        line_str = line_str.strip()
        cmp_result.append(json.loads(line_str))

    test_data_path = pipeline_config.eval_input_path
    if extract_data_func is not None:
      tmp_data_path = os.path.join(test_dir, 'pred_input_data')
      extract_data_func(test_data_path, tmp_data_path)
      test_data_path = tmp_data_path
    self._predict_and_check(
        test_data_path,
        export_dir_single,
        cmp_result,
        keys=keys,
        separator=separator)
    if test_multi:
      self._predict_and_check(
          test_data_path,
          export_dir_multi,
          cmp_result,
          keys=keys,
          separator=separator)
    test_utils.clean_up(test_dir)

  def _test_big_model_export(self,
                             pipeline_config_path,
                             test_data_path,
                             extract_data_func=None,
                             total_steps=50):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)

    lookup_op_path = os.path.join(easy_rec.ops_dir, 'libembed_op.so')
    tf.load_op_library(lookup_op_path)

    # prepare model
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir, total_steps=total_steps))

    test_utils.set_gpu_id(None)
    # the pipeline.config is produced by the prepare model cmd
    config_path = os.path.join(test_dir, 'pipeline.config')
    export_dir = os.path.join(test_dir, 'export/')
    export_cmd = """
      python -m easy_rec.python.export
        --pipeline_config_path %s
        --export_dir %s
        --asset_files %s
        --redis_url %s
        --redis_passwd %s
        --redis_threads 1
        --redis_write_kv 1
        --verbose 1
    """ % (config_path, export_dir, test_data_path, os.environ['redis_url'],
           os.environ['redis_passwd'])
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_%s.txt' % (test_dir, 'export'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    export_dir = gfile.Glob(export_dir + '[0-9][0-9][0-9]*')[0]
    _, test_data_name = os.path.split(test_data_path)
    assert gfile.Exists(export_dir + '/assets/' + test_data_name)

    # use checkpoint to prepare result
    result_path = os.path.join(test_dir, 'result.txt')
    predict_cmd = """
      python -m easy_rec.python.predict
        --pipeline_config_path %s
        --input_path %s
        --output_path %s
    """ % (config_path, test_data_path, result_path)
    proc = test_utils.run_cmd(predict_cmd % (),
                              '%s/log_%s.txt' % (test_dir, 'predict'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    with open(result_path, 'r') as fin:
      cmp_result = []
      for line_str in fin:
        line_str = line_str.strip()
        cmp_result.append(json.loads(line_str))

    if extract_data_func is not None:
      tmp_data_path = os.path.join(test_dir, 'pred_input_data')
      extract_data_func(test_data_path, tmp_data_path)
      test_data_path = tmp_data_path
    self._predict_and_check(test_data_path, export_dir, cmp_result)

  @unittest.skipIf(
      'redis_url' not in os.environ,
      'Only execute when redis is available: redis_url, redis_passwd')
  def test_big_model_export(self):
    pipeline_config_path = 'samples/model_config/multi_tower_export.config'
    test_data_path = 'data/test/export/data.csv'
    self._test_big_model_export(
        pipeline_config_path,
        test_data_path,
        extract_data_func=self._extract_data)

  @unittest.skipIf(
      'redis_url' not in os.environ,
      'Only execute when redis is available: redis_url, redis_passwd')
  def test_big_model_deepfm_export(self):
    pipeline_config_path = 'samples/model_config/deepfm_combo_on_avazu_ctr.config'
    test_data_path = 'data/test/dwd_avazu_ctr_deepmodel_10w.csv'
    self._test_big_model_export(
        pipeline_config_path,
        test_data_path,
        extract_data_func=self._extract_data)

  @unittest.skipIf(
      'redis_url' not in os.environ,
      'Only execute when redis is available: redis_url, redis_passwd')
  def test_big_model_din_export(self):
    pipeline_config_path = 'samples/model_config/din_on_taobao.config'
    test_data_path = 'data/test/tb_data/taobao_test_data'
    self._test_big_model_export(
        pipeline_config_path,
        test_data_path,
        extract_data_func=functools.partial(self._extract_data, offset=2))

  @unittest.skipIf(
      'redis_url' not in os.environ,
      'Only execute when redis is available: redis_url, redis_passwd')
  def test_big_model_wide_and_deep_export(self):
    pipeline_config_path = 'samples/model_config/wide_and_deep_two_opti.config'
    test_data_path = 'data/test/dwd_avazu_ctr_deepmodel_10w.csv'
    self._test_big_model_export(
        pipeline_config_path,
        test_data_path,
        extract_data_func=functools.partial(self._extract_data))

  @unittest.skipIf(
      'redis_url' not in os.environ or '-PAI' not in tf.__version__,
      'Only execute when pai-tf and redis is available: redis_url, redis_passwd'
  )
  def test_big_model_embedding_variable_export(self):
    pipeline_config_path = 'samples/model_config/taobao_fg_ev.config'
    test_data_path = 'data/test/rtp/taobao_valid_feature.txt'
    self._test_big_model_export(
        pipeline_config_path,
        test_data_path,
        self._extract_rtp_data,
        total_steps=1000)

  @unittest.skipIf(
      'oss_endpoint' not in os.environ or 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ or 'oss_path' not in os.environ or
      '-PAI' not in tf.__version__,
      'Only execute oss params(oss_endpoint,oss_ak,oss_sk) are specified,'
      'and pai-tf is available.')
  def test_big_model_embedding_variable_oss_export(self):
    pipeline_config_path = 'samples/model_config/taobao_fg_ev.config'
    test_data_path = 'data/test/rtp/taobao_valid_feature.txt'
    self._test_big_model_export_to_oss(
        pipeline_config_path,
        test_data_path,
        self._extract_rtp_data,
        total_steps=100)

  @unittest.skipIf(
      'oss_endpoint' not in os.environ or 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ or 'oss_path' not in os.environ or
      '-PAI' not in tf.__version__,
      'Only execute oss params(oss_endpoint,oss_ak,oss_sk) are specified,'
      'and pai-tf is available.')
  def test_big_model_embedding_variable_v2_oss_export(self):
    pipeline_config_path = 'samples/model_config/taobao_fg_ev_v2.config'
    test_data_path = 'data/test/rtp/taobao_valid_feature.txt'
    self._test_big_model_export_to_oss(
        pipeline_config_path,
        test_data_path,
        self._extract_rtp_data,
        total_steps=100)

  def _test_big_model_export_to_oss(self,
                                    pipeline_config_path,
                                    test_data_path,
                                    extract_data_func=None,
                                    total_steps=50):
    test_dir = test_utils.get_tmp_dir()
    logging.info('test dir: %s' % test_dir)

    lookup_op_path = os.path.join(easy_rec.ops_dir, 'libembed_op.so')
    tf.load_op_library(lookup_op_path)

    # prepare model
    self.assertTrue(
        test_utils.test_single_train_eval(
            pipeline_config_path, test_dir=test_dir, total_steps=total_steps))

    test_utils.set_gpu_id(None)
    # the pipeline.config is produced by the prepare model cmd
    config_path = os.path.join(test_dir, 'pipeline.config')
    export_dir = os.path.join(test_dir, 'export/')
    export_cmd = """
      python -m easy_rec.python.export
        --pipeline_config_path %s
        --export_dir %s
        --asset_files %s
        --oss_path %s
        --oss_endpoint %s
        --oss_ak %s --oss_sk %s
        --oss_threads 5
        --oss_timeout 10
        --oss_write_kv 1
        --verbose 1
    """ % (config_path, export_dir, test_data_path, os.environ['oss_path'],
           os.environ['oss_endpoint'], os.environ['oss_ak'],
           os.environ['oss_sk'])
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_%s.txt' % (test_dir, 'export'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    export_dir = gfile.Glob(export_dir + '[0-9][0-9][0-9]*')[0]
    _, test_data_name = os.path.split(test_data_path)
    assert gfile.Exists(export_dir + '/assets/' + test_data_name)

    # use checkpoint to prepare result
    result_path = os.path.join(test_dir, 'result.txt')
    predict_cmd = """
      python -m easy_rec.python.predict
        --pipeline_config_path %s
        --input_path %s
        --output_path %s
    """ % (config_path, test_data_path, result_path)
    proc = test_utils.run_cmd(predict_cmd,
                              '%s/log_%s.txt' % (test_dir, 'predict'))
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    with open(result_path, 'r') as fin:
      cmp_result = []
      for line_str in fin:
        line_str = line_str.strip()
        cmp_result.append(json.loads(line_str))

    if extract_data_func is not None:
      tmp_data_path = os.path.join(test_dir, 'pred_input_data')
      extract_data_func(test_data_path, tmp_data_path)
      test_data_path = tmp_data_path
    self._predict_and_check(test_data_path, export_dir, cmp_result)

  @unittest.skipIf(
      'oss_path' not in os.environ,
      'Only execute when oss is available: oss_path, oss_endpoint, oss_ak, oss_sk'
  )
  def test_big_model_export_to_oss(self):
    pipeline_config_path = 'samples/model_config/multi_tower_export.config'
    test_data_path = 'data/test/export/data.csv'
    self._test_big_model_export_to_oss(
        pipeline_config_path,
        test_data_path,
        extract_data_func=self._extract_data)

  @unittest.skipIf(
      'oss_path' not in os.environ,
      'Only execute when oss is available: oss_path, oss_endpoint, oss_ak, oss_sk'
  )
  def test_big_model_deepfm_export_to_oss(self):
    pipeline_config_path = 'samples/model_config/deepfm_combo_on_avazu_ctr.config'
    test_data_path = 'data/test/dwd_avazu_ctr_deepmodel_10w.csv'
    self._test_big_model_export_to_oss(
        pipeline_config_path,
        test_data_path,
        extract_data_func=self._extract_data)

  @unittest.skipIf(
      'oss_path' not in os.environ,
      'Only execute when oss is available: oss_path, oss_endpoint, oss_ak, oss_sk'
  )
  def test_big_model_din_export_to_oss(self):
    pipeline_config_path = 'samples/model_config/din_on_taobao.config'
    test_data_path = 'data/test/tb_data/taobao_test_data'
    self._test_big_model_export_to_oss(
        pipeline_config_path,
        test_data_path,
        extract_data_func=functools.partial(self._extract_data, offset=2))

  @unittest.skipIf(
      'oss_path' not in os.environ,
      'Only execute when oss is available: oss_path, oss_endpoint, oss_ak, oss_sk'
  )
  def test_big_model_wide_and_deep_export_to_oss(self):
    pipeline_config_path = 'samples/model_config/wide_and_deep_two_opti.config'
    test_data_path = 'data/test/dwd_avazu_ctr_deepmodel_10w.csv'
    self._test_big_model_export_to_oss(
        pipeline_config_path,
        test_data_path,
        extract_data_func=functools.partial(self._extract_data))


if __name__ == '__main__':
  tf.test.main()
