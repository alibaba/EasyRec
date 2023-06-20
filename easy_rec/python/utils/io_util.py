# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""IO utils.

isort:skip_file
"""
import logging
from future import standard_library
standard_library.install_aliases()

import os
import traceback
from subprocess import getstatusoutput

import six
import tensorflow as tf
from six.moves import http_client
from six.moves import urllib
import json
if six.PY2:
  from urllib import quote
else:
  from urllib.parse import quote

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

EASY_REC_RES_DIR = 'easy_rec_user_resources'
HTTP_MAX_NUM_RETRY = 5
HTTP_MAX_TIMEOUT = 600


def http_read(url, timeout=HTTP_MAX_TIMEOUT, max_retry=HTTP_MAX_NUM_RETRY):
  """Read data from url with maximum retry.

  Args:
    url: http url to be read
    timeout: specifies a timeout in seconds for blocking operations.
    max_retry: http max retry times.
  """
  num_read_try = 0
  data = None
  while num_read_try < max_retry:
    try:
      if six.PY2:
        url = url.encode('utf-8')
      url = quote(url, safe='%/:?=&')
      data = urllib.request.urlopen(url, timeout=timeout).read()
      break
    except http_client.IncompleteRead:
      tf.logging.warning('incomplete read exception, will retry: %s' % url)
      num_read_try += 1
    except Exception:
      tf.logging.error(traceback.format_exc())
      break

  if data is None:
    tf.logging.error('http read %s failed' % url)

  return data


def download(oss_or_url, dst_dir=''):
  """Download file.

  Args:
    oss_or_url: http or oss path
    dst_dir: destination directory
  Return:
    dst_file: local path for the downloaded file
  """
  _, basename = os.path.split(oss_or_url)
  if oss_or_url[:3] == 'oss':
    with tf.gfile.GFile(oss_or_url, 'rb') as infile:
      file_content = infile.read()
  elif oss_or_url[:4] == 'http':
    try:
      response = urllib.request.urlopen(oss_or_url, timeout=HTTP_MAX_TIMEOUT)
      file_content = response.read()
    except Exception as e:
      raise RuntimeError('Download %s failed: %s\n %s' %
                         (oss_or_url, str(e), traceback.format_exc()))
  else:
    tf.logging.warning('skip downloading %s, seems to be a local file' %
                       oss_or_url)
    return oss_or_url

  if dst_dir != '' and not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
  dst_file = os.path.join(dst_dir, basename)
  with tf.gfile.GFile(dst_file, 'wb') as ofile:
    ofile.write(file_content)

  return dst_file


def create_module_dir(dst_dir):
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    with open(os.path.join(dst_dir, '__init__.py'), 'w') as ofile:
      ofile.write('\n')


def download_resource(resource_path, dst_dir=EASY_REC_RES_DIR):
  """Download user resource.

  Args:
    resource_path: http or oss path
    dst_dir: destination directory
  """
  create_module_dir(dst_dir)
  _, basename = os.path.split(resource_path)
  if not basename.endswith('.py'):
    raise ValueError('resource %s should be python file' % resource_path)

  target = download(resource_path, dst_dir)

  return target


def download_and_uncompress_resource(resource_path, dst_dir=EASY_REC_RES_DIR):
  """Download user resource and uncompress it if necessary.

  Args:
    resource_path: http or oss path
    dst_dir: download destination directory
  """
  create_module_dir(dst_dir)

  _, basename = os.path.split(resource_path)
  if not basename.endswith('.tar.gz') and not basename.endswith('.zip') and \
     not basename.endswith('.py'):
    raise ValueError('resource %s should be tar.gz or zip or py' %
                     resource_path)

  download(resource_path, dst_dir)

  stat = 0
  if basename.endswith('tar.gz'):
    stat, output = getstatusoutput('cd %s && tar -zxf %s' % (dst_dir, basename))
  elif basename.endswith('zip'):
    stat, output = getstatusoutput('cd %s && unzip %s' % (dst_dir, basename))

  if stat != 0:
    raise ValueError('uncompress resoruce %s failed: %s' % resource_path,
                     output)

  return dst_dir


def oss_has_t_mode(target_file):
  """Test if current enviroment support t-mode written to oss."""
  if 'PAI' not in tf.__version__:
    return False
  # test if running on cluster
  test_file = target_file + '.tmp'
  try:
    with tf.gfile.GFile(test_file, 't') as ofile:
      ofile.write('a')
      pass
    tf.gfile.Remove(test_file)
    return True
  except:  # noqa: E722
    return False


def fix_oss_dir(path):
  """Make sure that oss dir endswith /."""
  if path.startswith('oss://') and not path.endswith('/'):
    return path + '/'
  return path


def save_data_to_json_path(json_path, data):
  with tf.gfile.GFile(json_path, 'w') as fout:
    fout.write(json.dumps(data))
  assert tf.gfile.Exists(json_path), 'in_save_data_to_json_path, save_failed'


def read_data_from_json_path(json_path):
  if json_path and tf.gfile.Exists(json_path):
    with tf.gfile.GFile(json_path, 'r') as fin:
      data = json.loads(fin.read())
    return data
  else:
    logging.info('json_path not exists, return None')
    return None
