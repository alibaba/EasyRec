# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.utils import config_util
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import fg_util
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils.check_utils import check_sequence, check_train_step

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)
tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_multi_string(
    'train_input_path', None, help='train data input path')
tf.app.flags.DEFINE_string(
    'edit_config_json',
    None,
    help='edit pipeline config str, example: {"model_dir":"experiments/",'
    '"feature_config.feature[0].boundaries":[4,5,6,7]}')
tf.app.flags.DEFINE_bool('is_on_ds', False, help='is on ds')

FLAGS = tf.app.flags.FLAGS


def _get_input_fn(data_config,
                  feature_configs,
                  data_path=None,
                  export_config=None):
  """Build estimator input function.

  Args:
    data_config:  dataset config
    feature_configs: FeatureConfig
    data_path: input_data_path
    export_config: configuration for exporting models,
      only used to build input_fn when exporting models

  Returns:
    subclass of Input
  """
  input_class_map = {y: x for x, y in data_config.InputType.items()}
  input_cls_name = input_class_map[data_config.input_type]

  input_class = Input.create_class(input_cls_name)
  task_id, task_num = estimator_utils.get_task_index_and_num()
  input_obj = input_class(
      data_config,
      feature_configs,
      data_path,
      task_index=task_id,
      task_num=task_num,
      check_mode=True
  )
  input_fn = input_obj.create_input(export_config)
  return input_fn

def _is_local(pipeline_config):
    input = pipeline_config.data_config.input_type
    if input in [DatasetConfig.InputType.OdpsInputV2, DatasetConfig.InputType.OdpsRTPInput]:
        return True
    elif input in [DatasetConfig.InputType.CSVInput, DatasetConfig.InputType.RTPInput]:
        return False
    else:
        assert False, "Currently only supports OdpsInputV2/OdpsRTPInput/CSVInput/RTPInput."

def _is_fg(pipeline_config):
    input = pipeline_config.data_config.input_type
    if input in [DatasetConfig.InputType.OdpsRTPInput, DatasetConfig.InputType.RTPInput]:
        return True
    else:
        return False


def check_env(pipeline_config):
    if pipeline_config.data_config.input_type not in [
        DatasetConfig.InputType.OdpsInputV2,
        DatasetConfig.InputType.OdpsRTPInput,
        DatasetConfig.InputType.CSVInput,
        DatasetConfig.InputType.RTPInput]:
        assert False, "Currently only supports OdpsInputV2/OdpsRTPInput/CSVInput/RTPInput."
    assert FLAGS.is_on_ds == _is_local(pipeline_config), \
        "If you run it in local/ds, the inputype should be CSVInput/RTPInput."

def update_pipeline(pipeline_config_path):
    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, False)

    if FLAGS.train_input_path:
        pipeline_config.train_input_path = ','.join(FLAGS.train_input_path)
        logging.info('update train_input_path to %s' %
                     pipeline_config.train_input_path)

    if pipeline_config.fg_json_path:
        fg_util.load_fg_json_to_config(pipeline_config)

    if FLAGS.edit_config_json:
        config_json = json.loads(FLAGS.edit_config_json)
        config_util.edit_config(pipeline_config, config_json)
    config_util.auto_expand_share_feature_configs(pipeline_config)
    return pipeline_config

def main(argv):
    assert FLAGS.pipeline_config_path, 'pipeline_config_path should not be empty when training!'
    pipeline_config = update_pipeline(FLAGS.pipeline_config_path)
    check_env(pipeline_config)
    check_train_step(pipeline_config)

    is_local = _is_local(pipeline_config)
    is_fg = _is_fg(pipeline_config)

    feature_configs = config_util.get_compatible_feature_configs(pipeline_config)
    logging.info("pipeline_config.train_input_path: %s" % pipeline_config.train_input_path)
    eval_input_fn = _get_input_fn(pipeline_config.data_config, feature_configs, pipeline_config.train_input_path)
    eval_spec = tf.estimator.EvalSpec(
        name='val',
        input_fn=eval_input_fn,
        steps=None,
        throttle_secs=10,
        exporters=[])
    input_iter = eval_spec.input_fn(
        mode=tf.estimator.ModeKeys.EVAL).make_one_shot_iterator()
    with tf.Session() as sess:
        try:
            while(True):
                input_feas, input_lbls = input_iter.get_next()
                features = sess.run(input_feas)
                check_sequence(pipeline_config, features)
        except tf.errors.OutOfRangeError:
            logging.info("pre-check finish...")
        except Exception as e:
            logging.error(str(e))

if __name__ == '__main__':
  tf.app.run()
