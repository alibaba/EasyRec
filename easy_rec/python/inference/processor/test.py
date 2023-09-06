# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import ctypes
import glob
import json
import logging
import os
import subprocess
import time

import numpy as np
from google.protobuf import text_format

from easy_rec.python.protos import dataset_pb2
from easy_rec.python.protos import pipeline_pb2
from easy_rec.python.protos import tf_predict_pb2

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

PROCESSOR_VERSION = 'LaRec-0.9.5d-b1b1604-TF-2.5.0-Linux'
PROCESSOR_FILE = PROCESSOR_VERSION + '.tar.gz'
PROCESSOR_URL = 'http://easyrec.oss-cn-beijing.aliyuncs.com/processor/' + PROCESSOR_FILE
PROCESSOR_ENTRY_LIB = 'processor/' + PROCESSOR_VERSION + '/larec/libtf_predictor.so'


def build_array_proto(array_proto, data, dtype):
  array_proto.array_shape.dim.append(len(data))

  if dtype == dataset_pb2.DatasetConfig.STRING:
    array_proto.string_val.extend([x.encode('utf-8') for x in data])
    array_proto.dtype = tf_predict_pb2.DT_STRING
  elif dtype == dataset_pb2.DatasetConfig.FLOAT:
    array_proto.float_val.extend([float(x) for x in data])
    array_proto.dtype = tf_predict_pb2.DT_FLOAT
  elif dtype == dataset_pb2.DatasetConfig.DOUBLE:
    array_proto.double_val.extend([float(x) for x in data])
    array_proto.dtype = tf_predict_pb2.DT_DOUBLE
  elif dtype == dataset_pb2.DatasetConfig.INT32:
    array_proto.int_val.extend([int(x) for x in data])
    array_proto.dtype = tf_predict_pb2.DT_INT32
  elif dtype == dataset_pb2.DatasetConfig.INT64:
    array_proto.int64_val.extend([np.int64(x) for x in data])
    array_proto.dtype = tf_predict_pb2.DT_INT64
  else:
    assert False, 'invalid datatype[%s]' % str(dtype)
  return array_proto


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_path', type=str, default=None, help='input data path')
  parser.add_argument(
      '--output_path', type=str, default=None, help='output data path')
  parser.add_argument(
      '--libc_path',
      type=str,
      default='/lib64/libc.so.6',
      help='libc.so.6 path')
  parser.add_argument(
      '--saved_model_dir', type=str, default=None, help='saved model directory')
  parser.add_argument(
      '--test_dir', type=str, default=None, help='test directory')
  args = parser.parse_args()

  if not os.path.exists('processor'):
    os.mkdir('processor')
  if not os.path.exists(PROCESSOR_ENTRY_LIB):
    if not os.path.exists('processor/' + PROCESSOR_FILE):
      subprocess.check_output(
          'wget %s -O processor/%s' % (PROCESSOR_URL, PROCESSOR_FILE),
          shell=True)
    subprocess.check_output(
        'cd processor && tar -zvxf %s' % PROCESSOR_FILE, shell=True)
    assert os.path.exists(
        PROCESSOR_ENTRY_LIB), 'invalid processor path: %s' % PROCESSOR_ENTRY_LIB

  assert os.path.exists(args.libc_path), '%s does not exist' % args.libc_path
  assert args.saved_model_dir is not None and os.path.isdir(
      args.saved_model_dir
  ), '%s is not a valid directory' % args.saved_model_dir
  assert args.input_path is not None and os.path.exists(
      args.input_path), '%s does not exist' % args.input_path
  assert args.output_path is not None, 'output_path is not set'

  pipeline_config = pipeline_pb2.EasyRecConfig()
  pipeline_config_path = os.path.join(args.saved_model_dir,
                                      'assets/pipeline.config')
  with open(pipeline_config_path) as fin:
    config_str = fin.read()
  text_format.Merge(config_str, pipeline_config)

  data_config = pipeline_config.data_config

  input_fields = [[]
                  for x in data_config.input_fields
                  if x.input_name not in data_config.label_fields]

  with open(args.input_path, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      line_toks = line_str.split(data_config.rtp_separator)[-1].split(chr(2))
      for i, tok in enumerate(line_toks):
        input_fields[i].append(tok)

  req = tf_predict_pb2.PredictRequest()
  req.signature_name = 'serving_default'
  for i in range(len(input_fields)):
    build_array_proto(req.inputs[data_config.input_fields[i + 1].input_name],
                      input_fields[i],
                      data_config.input_fields[i + 1].input_type)

  tf_predictor = ctypes.cdll.LoadLibrary(PROCESSOR_ENTRY_LIB)
  tf_predictor.saved_model_init.restype = ctypes.c_void_p
  handle = tf_predictor.saved_model_init(args.saved_model_dir.encode('utf-8'))
  logging.info('saved_model handle=%d' % handle)

  num_steps = pipeline_config.train_config.num_steps
  logging.info('num_steps=%d' % num_steps)

  # last_step could be greater than num_steps for sync_replicas: false
  train_dir = os.path.dirname(args.saved_model_dir.strip('/'))
  all_models = glob.glob(
      os.path.join(args.test_dir, 'train/model.ckpt-*.index'))
  iters = [int(x.split('-')[-1].replace('.index', '')) for x in all_models]
  iters.sort()
  last_step = iters[-1]
  logging.info('last_step=%d' % last_step)

  sparse_step = ctypes.c_int(0)
  dense_step = ctypes.c_int(0)
  start_ts = time.time()
  while sparse_step.value < last_step or dense_step.value < last_step:
    tf_predictor.saved_model_step(
        ctypes.c_void_p(handle), ctypes.byref(sparse_step),
        ctypes.byref(dense_step))
    time.sleep(1)
    if time.time() - start_ts > 300:
      logging.warning(
          'could not reach last_step, sparse_step=%d dense_step=%d' %
          (sparse_step.value, dense_step.value))
      break

  data_bin = req.SerializeToString()
  save_path = os.path.join(args.saved_model_dir, 'req.pb')
  with open(save_path, 'wb') as fout:
    fout.write(data_bin)
  logging.info('save request to %s' % save_path)

  tf_predictor.saved_model_predict.restype = ctypes.c_void_p
  out_len = ctypes.c_int(0)
  res_p = tf_predictor.saved_model_predict(
      ctypes.c_void_p(handle), data_bin, ctypes.c_int32(len(data_bin)),
      ctypes.byref(out_len))
  res_bytes = bytearray(ctypes.string_at(res_p, out_len))
  res = tf_predict_pb2.PredictResponse()
  res.ParseFromString(res_bytes)

  with open(args.output_path, 'w') as fout:
    logits = res.outputs['logits'].float_val
    probs = res.outputs['probs'].float_val
    for logit, prob in zip(logits, probs):
      fout.write(json.dumps({'logits': logit, 'probs': prob}) + '\n')

  # free memory
  tf_predictor.saved_model_release(ctypes.c_void_p(handle))
  libc = ctypes.cdll.LoadLibrary(args.libc_path)
  libc.free(ctypes.c_void_p(res_p))
