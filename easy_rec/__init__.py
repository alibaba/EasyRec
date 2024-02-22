# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import platform
import sys

from easy_rec.version import __version__

curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

# Avoid import tensorflow which conflicts with the version used in EasyRecProcessor
if 'PROCESSOR_TEST' not in os.environ:
  from tensorflow.python.platform import tf_logging
  # In DeepRec, logger.propagate of tf_logging is False, should be True
  tf_logging._logger.propagate = True

  def get_ops_dir():
    import tensorflow as tf
    if platform.system() == 'Linux':
      ops_dir = os.path.join(curr_dir, 'python/ops')
      if 'PAI' in tf.__version__:
        ops_dir = os.path.join(ops_dir, '1.12_pai')
      elif tf.__version__.startswith('1.12'):
        ops_dir = os.path.join(ops_dir, '1.12')
      elif tf.__version__.startswith('1.15'):
        if 'IS_ON_PAI' in os.environ:
          ops_dir = os.path.join(ops_dir, 'DeepRec')
        else:
          ops_dir = os.path.join(ops_dir, '1.15')
      else:
        tmp_version = tf.__version__.split('.')
        tmp_version = '.'.join(tmp_version[:2])
        ops_dir = os.path.join(ops_dir, tmp_version)
      return ops_dir
    else:
      return None

  ops_dir = get_ops_dir()
  if ops_dir is not None and not os.path.exists(ops_dir):
    logging.warning('ops_dir[%s] does not exist' % ops_dir)
    ops_dir = None

  from easy_rec.python.inference.predictor import Predictor  # isort:skip  # noqa: E402
  from easy_rec.python.main import evaluate  # isort:skip  # noqa: E402
  from easy_rec.python.main import distribute_evaluate  # isort:skip  # noqa: E402
  from easy_rec.python.main import export  # isort:skip  # noqa: E402
  from easy_rec.python.main import train_and_evaluate  # isort:skip  # noqa: E402
  from easy_rec.python.main import export_checkpoint  # isort:skip  # noqa: E402

  try:
    import tensorflow_io.oss
  except Exception:
    pass

  print('easy_rec version: %s' % __version__)
  print('Usage: easy_rec.help()')

_global_config = {}


def help():
  print("""
1 Train
1.1 Train 1gpu
  CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval
      --pipeline_config_path deepfm_combo_on_avazu_ctr.config
1.2 Train 2gpu
  sh scripts/train_2gpu.sh deepfm_combo_on_avazu_ctr.config
2 Eval
  CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.eval
      --pipeline_config_path deepfm_combo_on_avazu_ctr.config
3 Export
  CUDA_VISIBLE_DEVICES=""
    python -m easy_rec.python.export
      --pipeline_config_path deepfm_combo_on_avazu_ctr.config
      --export_dir models/export
4 Create config from excel
  python -m easy_rec.python.tools.create_config_from_excel
      --excel_path dwd_avazu_ctr_multi_tower.xls
      --output_path dwd_avazu_ctr_multi_tower.config
5. Inference:
  # use list input
  import csv
  from easy_rec.python.inference.predictor import Predictor
  predictor = Predictor(SAVED_MODEL_DIR)
  with open(INPUT_CSV, 'r') as fin:
    reader = csv.reader(fin)
    inputs = []
    for row in reader:
      inputs.append(row[1:])
    output_res = self._predictor.predict(inputs, batch_size=32)

  # use dict input
  import csv
  from easy_rec.python.inference.predictor import Predictor
  predictor = Predictor(SAVED_MODEL_DIR)
  field_keys = [ "field1", "field2", "field3", "field4", "field5",
                 "field6", "field7", "field8", "field9", "field10",
                 "field11", "field12", "field13", "field14", "field15",
                 "field16", "field17", "field18", "field19", "field20" ]
  with open(INPUT_CSV, 'r') as fin:
    reader = csv.reader(fin)
    inputs = []
    for row in reader:
      inputs.append({ f : row[fid+1] for fid, f in enumerate(field_keys) })
    output_res = self._predictor.predict(inputs, batch_size=32)
""")
