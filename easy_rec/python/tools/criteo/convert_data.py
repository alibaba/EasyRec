# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import gzip
import logging
import multiprocessing
import os
import traceback

import numpy as np
import pandas as pd
import six
from tensorflow.python.platform import gfile

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


def save_np_bin(labels, dense_arr, cate_arr, prefix):
  with gfile.GFile(prefix + '_label.bin', 'wb') as fout:
    fout.write(np.array(labels, dtype=np.int32).tobytes())
  with gfile.GFile(prefix + '_dense.bin', 'wb') as fout:
    fout.write(np.array(dense_arr, dtype=np.float32).tobytes())
  with gfile.GFile(prefix + '_category.bin', 'wb') as fout:
    fout.write(np.array(cate_arr, dtype=np.float32).tobytes())


def save_parquet(labels, dense_arr, cate_arr, prefix):
  df = {'is_click': labels}
  for i in range(1, 14):
    df['f' + str(i)] = dense_arr[:, i - 1]
  for i in range(1, 27):
    df['c' + str(i)] = cate_arr[:, i - 1]
  df = pd.DataFrame(df)
  save_path = prefix + '.parquet'
  logging.info('save to %s' % save_path)
  df.to_parquet(save_path)


def convert(input_path, prefix, part_record_num, save_format):
  logging.info('start to convert %s, part_record_num=%d, save_format=%s' %
               (input_path, part_record_num, save_format))
  save_func = save_np_bin
  if save_format == 'parquet':
    save_func = save_parquet
  batch_size = part_record_num
  labels = np.zeros([batch_size], dtype=np.int32)
  dense_arr = np.zeros([batch_size, 13], dtype=np.float32)
  cate_arr = np.zeros([batch_size, 26], dtype=np.uint32)
  part_id = 0
  total_line = 0
  try:
    sid = 0
    with gfile.GFile(input_path, 'rb') as gz_fin:
      for line_str in gzip.GzipFile(fileobj=gz_fin, mode='rb'):
        if six.PY3:
          line_str = str(line_str, 'utf-8')
        line_str = line_str.strip()
        line_toks = line_str.split('\t')
        labels[sid] = int(line_toks[0])

        for j in range(1, 14):
          x = line_toks[j]
          dense_arr[sid, j - 1] = float(x) if x != '' else 0.0

        for j in range(14, 40):
          x = line_toks[j]
          cate_arr[sid, j - 14] = int(x, 16) if x != '' else 0

        sid += 1
        if sid == batch_size:
          save_func(labels, dense_arr, cate_arr, prefix + '_' + str(part_id))
          logging.info('\t%s write part: %d' % (input_path, part_id))
          part_id += 1
          total_line += sid
          sid = 0
    if sid > 0:
      save_func(labels[:sid], dense_arr[:sid], cate_arr[:sid],
                prefix + '_' + str(part_id))
      logging.info('\t%s write final part: %d' % (input_path, part_id))
      part_id += 1
      total_line += sid
  except Exception as ex:
    logging.error('convert %s failed: %s' % (input_path, str(ex)))
    logging.error(traceback.format_exc())
    return
  logging.info('done convert %s, total_line=%d, part_num=%d' %
               (input_path, total_line, part_id))


if __name__ == '__main__':
  """Convert criteo 1T data to binary format.

  The outputs are stored in multiple parts, each with at most part_record_num samples.
  Each part consists of 3 files:
      xxx_yyy_label.bin,
      xxx_yyy_dense.bin,
      xxx_yyy_category.bin,
  xxx is in range [0-23], range of yyy is determined by part_record_num,

  If part_record_num is set to the default value 8M, there will be 535 parts. We convert
  the data on machine with 64GB memory, if you memory is limited, you can convert the .gz
  files one by one, or you can set a small part_record_num.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_dir', type=str, default=None, help='criteo 1t data dir')
  parser.add_argument(
      '--save_dir',
      type=str,
      default=None,
      help='criteo binary data output dir ')
  parser.add_argument(
      '--save_format',
      type=str,
      default='npy',
      help='save format, choices: npy|parquet')
  parser.add_argument(
      '--part_record_num',
      type=int,
      default=1024 * 1024 * 8,
      help='the maximal number of samples in each binary file')
  parser.add_argument(
      '--dt',
      nargs='*',
      type=int,
      help='select days to convert, default to select all: 0-23')

  args = parser.parse_args()

  assert args.input_dir, 'input_dir is not set'
  assert args.save_dir, 'save_dir is not set'

  save_dir = args.save_dir
  if not save_dir.endswith('/'):
    save_dir = save_dir + '/'
  if not gfile.IsDirectory(save_dir):
    gfile.MakeDirs(save_dir)

  if args.dt is None or len(args.dt) == 0:
    days = list(range(0, 24))
  else:
    days = list(args.dt)

  proc_arr = []
  for d in days:
    input_path = os.path.join(args.input_dir, 'day_%d.gz' % d)
    prefix = os.path.join(args.save_dir, str(d))
    proc = multiprocessing.Process(
        target=convert,
        args=(input_path, prefix, args.part_record_num, args.save_format))
    convert(input_path, prefix, args.part_record_num, args.save_format)
    proc.start()
    proc_arr.append(proc)
  for proc in proc_arr:
    proc.join()
