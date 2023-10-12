# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import logging
import os
import shutil
import sys

import oss2

from easy_rec.python.test.odps_test_util import OdpsOSSConfig
from easy_rec.python.test.odps_test_util import get_oss_bucket


def download_data(ali_bucket, script_path):
  """Download data from alibaba bucket(readonly) to script_path.

  Args:
    ali_bucket: alibaba bucket
    script_path: down path
  """
  if not os.path.exists(script_path):
    raise '%s path is not exists' % script_path
  if os.path.exists(os.path.join(script_path, 'test')):
    shutil.rmtree(os.path.join(script_path, 'test'))

  # download data from oss://${ali_bucket}/data/odps_test/
  # to script_path/test_data
  for obj in oss2.ObjectIterator(ali_bucket, prefix='data/odps_test/'):
    obj_key = obj.key
    tmp_oss_dir = os.path.split(obj_key)[0]
    obj_path = os.path.join(script_path, tmp_oss_dir)
    try:
      os.makedirs(obj_path)
    except Exception:
      pass

    if obj_key.endswith('/'):
      continue

    dst_name = obj_key.replace('data/odps_test/', 'test_data/')
    dst_path = os.path.join(script_path, dst_name)
    dst_dir, _ = os.path.split(dst_path)
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    ali_bucket.get_object_to_file(obj_key, dst_path)
    logging.info('down file oss://%s/%s to %s completed' %
                 (ali_bucket.bucket_name, obj_key, dst_path))


def merge_files(merge_dir, merge_out):
  """Merge files in merge_dir into merge_out.

  Args:
    merge_dir: files of directory to merge.
    merge_out: merged output file.
  """
  input_files = list(glob.glob(merge_dir + '/*'))
  logging.info('merge %s into %s' % (','.join(input_files), merge_out))
  with open(merge_out, 'w') as fout:
    for input_path in glob.glob(merge_dir + '/*'):
      with open(input_path, 'r') as fin:
        for line_str in fin:
          fout.write(line_str)
  return merge_out


def change_files(odps_oss_config, file_path):
  """Update params in config files.

  Args:
    odps_oss_config: odps oss test config
    file_path: config file path
  """
  # skip temporary file
  if len(file_path) > 3 and file_path[-4:-1] == '.sw':
    return

  endpoint = odps_oss_config.endpoint.replace('http://', '')
  # endpoint_internal = endpoint.replace('.aliyuncs.com',
  #                                      '-internal.aliyuncs.com')

  test_data_dir = os.path.join(odps_oss_config.temp_dir, 'test_data')

  with open(file_path, 'r') as fin:
    lines = fin.readlines()

  with open(file_path, 'w') as fw:
    for line in lines:
      if 'pai' in line.lower() and 'easy_rec_ext' in line.lower():
        line = 'pai -name ' + odps_oss_config.algo_name + '\n'
        if odps_oss_config.algo_project:
          line += '-project=%s\n' % odps_oss_config.algo_project
        if odps_oss_config.algo_res_project:
          line += '-Dres_project=%s\n' % odps_oss_config.algo_res_project
        if odps_oss_config.algo_version:
          line += '-Dversion=%s\n' % odps_oss_config.algo_version

      if odps_oss_config.is_outer:
        line = line.replace('{OSS_BUCKET_NAME}', odps_oss_config.bucket_name)
        line = line.replace('{ROLEARN}', odps_oss_config.arn)
        line = line.replace('{OSS_ENDPOINT}', endpoint)
      else:
        tmp_e = odps_oss_config.endpoint
        # tmp_e = tmp_e.replace('oss-cn-', 'cn-')
        # tmp_e = tmp_e.replace('.aliyuncs.com', '.oss-internal.aliyun-inc.com')
        if '-Dbuckets=' in line:
          line = '-Dbuckets=oss://%s/?role_arn=%s&host=%s\n' % (
              odps_oss_config.bucket_name, odps_oss_config.arn, tmp_e)
        elif '-Darn=' in line or '-DossHost' in line:
          continue
        line = line.replace('{OSS_BUCKET_NAME}', odps_oss_config.bucket_name)

      line = line.replace('{TIME_STAMP}', str(odps_oss_config.time_stamp))

      if 'tunnel upload' in line:
        line = line.replace('{TEST_DATA_DIR}', test_data_dir)
        # merge files
        toks = [x for x in line.split(' ') if x != '']
        merge_path = toks[2]
        merge_dir = '_'.join(merge_path.split('_')[:-1])
        if not os.path.exists(merge_path):
          merge_files(merge_dir, merge_path)

      # for emr odps test only
      line = line.replace('{TEMP_DIR}', str(odps_oss_config.temp_dir))
      line = line.replace('{ODPS_PROJ_NAME}', odps_oss_config.project_name)
      line = line.replace('{EXP_NAME}', odps_oss_config.exp_dir)
      fw.write(line)


def put_data_to_bucket(odps_oss_config):
  """Upload test data and configs to my_bucket.

  Args:
    odps_oss_config: odps oss config obj
  """
  test_bucket = get_oss_bucket(odps_oss_config.oss_key,
                               odps_oss_config.oss_secret,
                               odps_oss_config.endpoint,
                               odps_oss_config.bucket_name)
  for sub_dir in ['configs']:
    for root, dirs, files in os.walk(
        os.path.join(odps_oss_config.temp_dir, sub_dir)):
      for one_file in files:
        file_path = os.path.join(root, one_file)
        obj_path = file_path.split(sub_dir + '/')[1]
        dst_path = os.path.join(odps_oss_config.exp_dir, sub_dir, obj_path)
        test_bucket.put_object_from_file(dst_path, file_path)
        logging.info('put %s to oss://%s/%s' %
                     (file_path, odps_oss_config.bucket_name, dst_path))


def prepare(odps_oss_config):
  logging.info('temp_dir = %s' % odps_oss_config.temp_dir)
  ali_bucket = get_oss_bucket(odps_oss_config.ali_oss_key,
                              odps_oss_config.ali_oss_secret,
                              odps_oss_config.ali_bucket_endpoint,
                              odps_oss_config.ali_bucket_name)
  shutil.copytree(odps_oss_config.script_path, odps_oss_config.temp_dir)
  logging.info('start down data')
  download_data(ali_bucket, odps_oss_config.temp_dir)
  logging.info('down data completed')

  # update params in config files
  for root, dirs, files in os.walk(odps_oss_config.temp_dir):
    for file in files:
      file_path = os.path.join(root, file)
      # drop .template
      if file_path.endswith('.template'):
        tmp_path = file_path[:-len('.template')]
        os.rename(file_path, tmp_path)
        file_path = tmp_path
      if 'data' not in file_path:
        logging.info('modify %s' % file_path)
        change_files(odps_oss_config, file_path)

  # put data to test bucket
  put_data_to_bucket(odps_oss_config)

  # mkdir log dir
  if not os.path.exists(odps_oss_config.log_dir):
    os.makedirs(odps_oss_config.log_dir)


if __name__ == '__main__':
  if len(sys.argv) < 5:
    print('usage: %s ossutilconfig bucket_name rolearn odpsconfig' %
          sys.argv[0])
    sys.exit(1)

  odps_oss_config = OdpsOSSConfig()
  odps_oss_config.load_oss_config(sys.argv[1])
  odps_oss_config.bucket_name = sys.argv[2]
  odps_oss_config.arn = sys.argv[3]
  odps_oss_config.load_odps_config(sys.argv[4])
  prepare(odps_oss_config)
