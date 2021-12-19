# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.utils import config_util, fg_util, test_utils

if tf.__version__ >= '2.0':
    tf = tf.compat.v1


class FGTest(tf.test.TestCase):
    def __init__(self, methodName='FGTest'):
        super(FGTest, self).__init__(methodName=methodName)

    def setUp(self):
        logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        self._test_dir = test_utils.get_tmp_dir()
        self._success = True
        logging.info('test dir: %s' % self._test_dir)

    def tearDown(self):
        test_utils.set_gpu_id(None)
        if self._success:
            test_utils.clean_up(self._test_dir)

    def test_fg_json_to_config(self):
        pipeline_config_path = 'samples/rtp_fg/fg_test_extensions.config'
        final_pipeline_config_path = 'samples/rtp_fg/fg_test_extensions_final.config'
        fg_path = 'samples/rtp_fg/fg_test_extensions.json'

        pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        pipeline_config.fg_json_path = fg_path
        fg_util.load_fg_json_to_config(pipeline_config)
        pipeline_config_str = text_format.MessageToString(pipeline_config, as_utf8=True)

        final_pipeline_config = config_util.get_configs_from_pipeline_file(final_pipeline_config_path)
        final_pipeline_config_str = text_format.MessageToString(final_pipeline_config, as_utf8=True)
        self.assertEqual(pipeline_config_str, final_pipeline_config_str)


if __name__ == '__main__':
    tf.test.main()
