# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf
from google.protobuf import text_format

from easy_rec.python.utils import config_util
from easy_rec.python.utils import fg_util


class FGTest(tf.test.TestCase):

  def __init__(self, methodName='FGTest'):
    super(FGTest, self).__init__(methodName=methodName)
    self._pipeline_config_path = 'samples/rtp_fg/fg_test_extensions.config'
    self._final_pipeline_config_path = 'samples/rtp_fg/fg_test_extensions_final.config'
    self._fg_path = 'samples/rtp_fg/fg_test_extensions.json'

  def test_fg_json_to_config(self):
    pipeline_config = config_util.get_configs_from_pipeline_file(
        self._pipeline_config_path)
    pipeline_config.fg_json_path = self._fg_path
    fg_util.load_fg_json_to_config(pipeline_config)
    pipeline_config_str = text_format.MessageToString(
        pipeline_config, as_utf8=True)

    final_pipeline_config = config_util.get_configs_from_pipeline_file(
        self._final_pipeline_config_path)
    final_pipeline_config_str = text_format.MessageToString(
        final_pipeline_config, as_utf8=True)
    self.assertEqual(pipeline_config_str, final_pipeline_config_str)


if __name__ == '__main__':
  tf.test.main()
