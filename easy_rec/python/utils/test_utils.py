# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contains functions which are convenient for unit testing.

isort:skip_file
"""
from future import standard_library
standard_library.install_aliases()
import yaml
import glob
import json
import logging
import os
import random
import shutil
import string
import subprocess
import time
import six
from multiprocessing import Process
from subprocess import getstatusoutput
from tensorflow.python.platform import gfile
import numpy as np
from easy_rec.python.protos.train_pb2 import DistributionStrategy
from easy_rec.python.utils import config_util
from easy_rec.python.protos.pipeline_pb2 import EasyRecConfig
from easy_rec.python.utils.io_util import read_data_from_json_path
from easy_rec.python.utils import constant

TEST_DIR = './tmp/easy_rec_test'

# parallel run of tests could take more time
TEST_TIME_OUT = int(os.environ.get('TEST_TIME_OUT', 1800))


def get_hdfs_tmp_dir(test_dir):
  """Create a randomly of directory  in HDFS."""
  tmp_name = ''.join(
      [random.choice(string.ascii_letters + string.digits) for i in range(8)])
  assert isinstance(test_dir, str)
  test_rand_dir = os.path.join(test_dir, tmp_name)
  gfile.MkDir(test_rand_dir)
  return test_rand_dir


def proc_wait(proc, timeout=1200):
  t0 = time.time()
  while proc.poll() is None and time.time() - t0 < timeout:
    time.sleep(1)
  if proc.poll() is None:
    logging.warning('proc[pid=%d] timeout[%d], will kill the proc' %
                    (proc.pid, timeout))
    proc.terminate()
  while proc.poll() is None:
    time.sleep(1)


def get_tmp_dir():
  max_retry = 5
  while max_retry > 0:
    tmp_name = ''.join([
        random.choice(string.ascii_letters + string.digits) for i in range(12)
    ])
    if os.environ.get('TEST_DIR', '') != '':
      global TEST_DIR
      TEST_DIR = os.environ['TEST_DIR']
    dir_name = os.path.join(TEST_DIR, tmp_name)
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
      return dir_name
    else:
      max_retry -= 1
  raise RuntimeError('Failed to get_tmp_dir: max_retry=%d' % max_retry)


def clear_all_tmp_dirs():
  shutil.rmtree(TEST_DIR)


def set_gpu_id(gpu_id_str):
  env = os.environ
  if gpu_id_str is None:
    env['CUDA_VISIBLE_DEVICES'] = ''
  else:
    env['CUDA_VISIBLE_DEVICES'] = gpu_id_str


def get_available_gpus():
  if 'TEST_DEVICES' in os.environ:
    gpus = os.environ['TEST_DEVICES'].split(',')
  else:
    gpus = glob.glob('/dev/nvidia[0-9]*')
    gpus = [gpu.replace('/dev/nvidia', '') for gpu in gpus]
  logging.info('available gpus %s' % gpus)
  return gpus


def run_cmd(cmd_str, log_file, env=None):
  """Run a shell cmd."""
  cmd_str = cmd_str.replace('\r', ' ').replace('\n', ' ')
  logging.info('RUNCMD: %s > %s 2>&1 ' % (cmd_str, log_file))
  with open(log_file, 'w') as lfile:
    proc = subprocess.Popen(
        cmd_str, stdout=lfile, stderr=subprocess.STDOUT, shell=True, env=env)
    if six.PY2:
      # for debug purpose
      proc.args = cmd_str
    return proc


def RunAsSubprocess(f):
  """Function dectorator to run function in subprocess.

  if a function will start a tf session. Because tensorflow gpu memory will not be cleared until the
  process exit.
  """

  def wrapped_f(*args, **kw):
    p = Process(target=f, args=args, kwargs=kw)
    p.start()
    p.join(timeout=600)
    assert p.exitcode == 0, 'subprocess run failed: %s' % f.__name__

  return wrapped_f


def clean_up(test_dir):
  if test_dir is not None:
    shutil.rmtree(test_dir)

  # reset to cpu mode
  set_gpu_id(None)


def clean_up_hdfs(test_dir):
  if gfile.Exists(test_dir):
    gfile.DeleteRecursively(test_dir)
  set_gpu_id(None)


def _replace_data_for_test(data_path):
  """Replace real data with test data."""
  test_data = {}

  change = False
  releated_datasets = []
  for k, config in test_data.items():
    if k in data_path:
      releated_datasets.append(k)

  # if there are multiple keyword detected, use the longest one
  if len(releated_datasets) > 0:
    score = [len(k) for k in releated_datasets]
    best_match = np.argmax(score)
    data_path = test_data[releated_datasets[best_match]]

    change = True
  assert change, 'Failed to replace data with test data'

  return data_path


def _load_config_for_test(pipeline_config_path,
                          test_dir,
                          total_steps=50,
                          num_epochs=0):
  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  train_config = pipeline_config.train_config
  eval_config = pipeline_config.eval_config
  data_config = pipeline_config.data_config

  train_config.num_steps = total_steps
  # change model_dir
  pipeline_config.model_dir = os.path.join(test_dir, 'train')
  logging.info('test_model_dir %s' % pipeline_config.model_dir)
  eval_config.num_examples = max(10, data_config.batch_size)
  data_config.num_epochs = num_epochs
  return pipeline_config


def _load_config_for_distribute_eval(pipeline_config_path, test_dir):
  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  pipeline_config.model_dir = test_dir
  logging.info('test_model_dir %s' % pipeline_config.model_dir)
  return pipeline_config


def test_datahub_train_eval(pipeline_config_path,
                            odps_oss_config,
                            test_dir,
                            process_pipeline_func=None,
                            total_steps=50,
                            post_check_func=None):
  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)

  if not isinstance(pipeline_config_path, EasyRecConfig):
    logging.info('testing pipeline config %s' % pipeline_config_path)
  if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']

  if isinstance(pipeline_config_path, EasyRecConfig):
    pipeline_config = pipeline_config_path
  else:
    pipeline_config = _load_config_for_test(pipeline_config_path, test_dir,
                                            total_steps)

  pipeline_config.train_config.train_distribute = 0
  pipeline_config.train_config.num_gpus_per_worker = 1
  pipeline_config.train_config.sync_replicas = False

  pipeline_config.datahub_train_input.akId = odps_oss_config.dh_id
  pipeline_config.datahub_train_input.akSecret = odps_oss_config.dh_key
  pipeline_config.datahub_train_input.region = odps_oss_config.dh_endpoint
  pipeline_config.datahub_train_input.project = odps_oss_config.dh_project
  pipeline_config.datahub_train_input.topic = odps_oss_config.dh_topic

  pipeline_config.datahub_eval_input.akId = odps_oss_config.dh_id
  pipeline_config.datahub_eval_input.akSecret = odps_oss_config.dh_key
  pipeline_config.datahub_eval_input.region = odps_oss_config.dh_endpoint
  pipeline_config.datahub_eval_input.project = odps_oss_config.dh_project
  pipeline_config.datahub_eval_input.topic = odps_oss_config.dh_topic

  if process_pipeline_func is not None:
    assert callable(process_pipeline_func)
    pipeline_config = process_pipeline_func(pipeline_config)
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')
  train_cmd = 'python -m easy_rec.python.train_eval --pipeline_config_path %s' % \
      test_pipeline_config_path
  proc = run_cmd(train_cmd, '%s/log_%s.txt' % (test_dir, 'master'))
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.warning(
        'train %s failed[pid=%d][code=%d][args=%s]' %
        (test_pipeline_config_path, proc.pid, proc.returncode, proc.args))
    return False
  if post_check_func:
    return post_check_func(pipeline_config)
  return True


def _Load_config_for_test_eval(pipeline_config_path):
  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  return pipeline_config


def test_single_train_eval(pipeline_config_path,
                           test_dir,
                           process_pipeline_func=None,
                           hyperparam_str='',
                           total_steps=50,
                           post_check_func=None,
                           check_mode=False,
                           fine_tune_checkpoint=None,
                           extra_cmd_args=None,
                           timeout=-1):
  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)

  if not isinstance(pipeline_config_path, EasyRecConfig):
    logging.info('testing pipeline config %s' % pipeline_config_path)
  if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']

  if isinstance(pipeline_config_path, EasyRecConfig):
    pipeline_config = pipeline_config_path
  else:
    pipeline_config = _load_config_for_test(pipeline_config_path, test_dir,
                                            total_steps)

  pipeline_config.train_config.train_distribute = 0
  pipeline_config.train_config.num_gpus_per_worker = 1
  pipeline_config.train_config.sync_replicas = False
  if process_pipeline_func is not None:
    assert callable(process_pipeline_func)
    pipeline_config = process_pipeline_func(pipeline_config)
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')
  train_cmd = 'python -m easy_rec.python.train_eval --pipeline_config_path=' + test_pipeline_config_path
  if hyperparam_str:
    train_cmd += ' --edit_config_json=\'%s\'' % hyperparam_str
  if fine_tune_checkpoint:
    train_cmd += ' --fine_tune_checkpoint %s' % fine_tune_checkpoint
  if check_mode:
    train_cmd += ' --check_mode'
  if extra_cmd_args:
    train_cmd += ' '
    train_cmd += extra_cmd_args
  proc = run_cmd(train_cmd, '%s/log_%s.txt' % (test_dir, 'master'))
  proc_wait(proc, timeout=TEST_TIME_OUT if timeout < 0 else timeout)
  if proc.returncode != 0:
    logging.error('train %s failed' % test_pipeline_config_path)
    return False
  if post_check_func:
    return post_check_func(pipeline_config)
  return True


def test_single_pre_check(pipeline_config_path, test_dir):
  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)

  if not isinstance(pipeline_config_path, EasyRecConfig):
    logging.info('testing pipeline config %s' % pipeline_config_path)
  if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']

  if isinstance(pipeline_config_path, EasyRecConfig):
    pipeline_config = pipeline_config_path
  else:
    pipeline_config = _load_config_for_test(pipeline_config_path, test_dir)

  pipeline_config.train_config.train_distribute = 0
  pipeline_config.train_config.num_gpus_per_worker = 1
  pipeline_config.train_config.sync_replicas = False

  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')
  train_cmd = 'python -m easy_rec.python.tools.pre_check --pipeline_config_path %s ' % (
      test_pipeline_config_path)

  proc = run_cmd(train_cmd, '%s/log_%s.txt' % (test_dir, 'master'))
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.error('train %s failed' % test_pipeline_config_path)
    return False
  return True


def test_single_predict(test_dir, input_path, output_path, saved_model_dir):
  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)

  predict_cmd = 'python -m easy_rec.python.predict --input_path %s --output_path %s --saved_model_dir %s' % (
      input_path, output_path, saved_model_dir)

  proc = run_cmd(predict_cmd, '%s/log_%s.txt' % (test_dir, 'master'))
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.error('predict failed')
    return False
  return True


def test_feature_selection(pipeline_config):
  model_dir = pipeline_config.model_dir
  pipeline_config_path = os.path.join(model_dir, 'pipeline.config')
  output_dir = os.path.join(model_dir, 'feature_selection')
  cmd = 'python -m easy_rec.python.tools.feature_selection --config_path %s ' \
        '--output_dir %s --topk 5 --visualize true' % (pipeline_config_path, output_dir)
  proc = run_cmd(cmd, os.path.join(model_dir, 'log_feature_selection.txt'))
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.error('feature selection %s failed' % pipeline_config_path)
    return False
  return True


def yaml_replace(train_yaml_path,
                 pipline_config_path,
                 test_pipeline_config_path,
                 test_export_dir=None):
  with open(train_yaml_path, 'r', encoding='utf-8') as _file:
    sample = _file.read()
    x = yaml.load(sample)
    _command = x['app']['command']
    if test_export_dir is not None:
      _command = _command.replace(pipline_config_path,
                                  test_pipeline_config_path).replace(
                                      '{EXPOERT_DIR}', test_export_dir)
    else:
      _command = _command.replace(pipline_config_path,
                                  test_pipeline_config_path)
    x['app']['command'] = _command

  with open(train_yaml_path, 'w', encoding='utf-8') as _file:
    yaml.dump(x, _file)


def test_hdfs_train_eval(pipeline_config_path,
                         train_yaml_path,
                         test_dir,
                         process_pipeline_func=None,
                         hyperparam_str='',
                         total_steps=2000):

  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)
  logging.info('testing pipeline config %s' % pipeline_config_path)
  logging.info('train_yaml_path %s' % train_yaml_path)
  if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']
  pipeline_config = _load_config_for_test(pipeline_config_path, test_dir,
                                          total_steps)
  logging.info('model_dir in pipeline_config has been modified')
  pipeline_config.train_config.train_distribute = 0
  pipeline_config.train_config.num_gpus_per_worker = 1
  pipeline_config.train_config.sync_replicas = False
  if process_pipeline_func is not None:
    assert callable(process_pipeline_func)
    pipeline_config = process_pipeline_func(pipeline_config)
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')
  yaml_replace(train_yaml_path, pipeline_config_path, test_pipeline_config_path)
  logging.info('test_pipeline_config_path is %s' % test_pipeline_config_path)
  train_cmd = 'el_submit -yaml %s' % train_yaml_path
  proc = subprocess.Popen(train_cmd.split(), stderr=subprocess.STDOUT)
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.error('train %s failed' % test_pipeline_config_path)
    logging.error('train_yaml %s failed' % train_yaml_path)
  return proc.returncode == 0


def test_hdfs_eval(pipeline_config_path,
                   eval_yaml_path,
                   test_dir,
                   process_pipeline_func=None,
                   hyperparam_str=''):

  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)
  logging.info('testing export pipeline config %s' % pipeline_config_path)
  logging.info('eval_yaml_path %s' % eval_yaml_path)
  if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']
  pipeline_config = _Load_config_for_test_eval(pipeline_config_path)
  if process_pipeline_func is not None:
    assert callable(process_pipeline_func)
    pipeline_config = process_pipeline_func(pipeline_config)
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')
  yaml_replace(eval_yaml_path, pipeline_config_path, test_pipeline_config_path)
  logging.info('test_pipeline_config_path is %s' % test_pipeline_config_path)
  eval_cmd = 'el_submit -yaml %s' % eval_yaml_path
  proc = subprocess.Popen(eval_cmd.split(), stderr=subprocess.STDOUT)
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.error('eval %s failed' % test_pipeline_config_path)
    logging.error('eval_yaml %s failed' % eval_yaml_path)
  return proc.returncode == 0


def test_hdfs_export(pipeline_config_path,
                     export_yaml_path,
                     test_dir,
                     process_pipeline_func=None,
                     hyperparam_str=''):

  gpus = get_available_gpus()
  if len(gpus) > 0:
    set_gpu_id(gpus[0])
  else:
    set_gpu_id(None)
  logging.info('testing export pipeline config %s' % pipeline_config_path)
  logging.info('export_yaml_path %s' % export_yaml_path)
  if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']
  pipeline_config = _Load_config_for_test_eval(pipeline_config_path)
  if process_pipeline_func is not None:
    assert callable(process_pipeline_func)
    pipeline_config = process_pipeline_func(pipeline_config)
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')
  test_export_path = os.path.join(test_dir, 'export_dir')
  yaml_replace(export_yaml_path, pipeline_config_path,
               test_pipeline_config_path, test_export_path)
  logging.info('test_pipeline_config_path is %s' % test_pipeline_config_path)
  eval_cmd = 'el_submit -yaml %s' % export_yaml_path
  proc = subprocess.Popen(eval_cmd.split(), stderr=subprocess.STDOUT)
  proc_wait(proc, timeout=TEST_TIME_OUT)
  if proc.returncode != 0:
    logging.error('export %s failed' % test_pipeline_config_path)
    logging.error('export_yaml %s failed' % export_yaml_path)
  return proc.returncode == 0


def _ports_in_use(ports):
  ports_str = ''
  for i, port in enumerate(ports):
    if i > 0:
      ports_str += '|'
    ports_str += '0.0.0.0:%d|127.0.0.1:%d' % (port, port)
  stat, output = getstatusoutput('netstat -tlnp | grep -E %s' % ports_str)
  return stat == 0


def get_ports_base(num_worker):
  port_base = int(os.environ.get('PORT_BASE', 10000))
  num_try = 10
  for i in range(num_try):
    ports = np.random.randint(port_base, port_base + 5000, size=num_worker)
    if not _ports_in_use(ports):
      return ports
    logging.info('ports %s in use, retry...' % ports)


def _get_ports(num_worker):
  # port queue to deals with port conflicts when multiple
  # test cases run in parallel
  if 'ports' in os.environ:
    ports = os.environ['ports']
    port_arr = [int(x) for x in ports.split(',')]
    assert len(port_arr) >= num_worker, 'not enough ports: %s, required: %d'\
        % (ports, num_worker)
    return port_arr[:num_worker]
  else:
    return get_ports_base(num_worker)


def _ps_worker_train(pipeline_config_path,
                     test_dir,
                     num_worker,
                     num_evaluator=0,
                     fit_on_eval=False,
                     fit_on_eval_steps=None):
  gpus = get_available_gpus()
  # not enough gpus, run on cpu only
  if len(gpus) < num_worker:
    gpus = [None] * num_worker
  ports = _get_ports(num_worker + 1)
  chief_or_master = 'master' if num_evaluator == 0 else 'chief'
  cluster = {
      chief_or_master: ['localhost:%d' % ports[0]],
      'worker': ['localhost:%d' % ports[i] for i in range(1, num_worker)],
      'ps': ['localhost:%d' % ports[-1]]
  }
  tf_config = {'cluster': cluster}
  procs = {}
  tf_config['task'] = {'type': chief_or_master, 'index': 0}
  os.environ['TF_CONFIG'] = json.dumps(tf_config)
  set_gpu_id(gpus[0])
  train_cmd = 'python -m easy_rec.python.train_eval --pipeline_config_path %s' % pipeline_config_path
  if fit_on_eval:
    train_cmd += ' --fit_on_eval'
    if fit_on_eval_steps is not None:
      train_cmd += ' --fit_on_eval_steps ' + str(int(fit_on_eval_steps))
  procs[chief_or_master] = run_cmd(
      train_cmd, '%s/log_%s.txt' % (test_dir, chief_or_master))
  tf_config['task'] = {'type': 'ps', 'index': 0}
  os.environ['TF_CONFIG'] = json.dumps(tf_config)
  set_gpu_id('')
  procs['ps'] = run_cmd(train_cmd, '%s/log_%s.txt' % (test_dir, 'ps'))

  for idx in range(num_worker - 1):
    tf_config['task'] = {'type': 'worker', 'index': idx}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    set_gpu_id(gpus[idx + 1])
    worker_name = 'worker_%d' % idx
    procs[worker_name] = run_cmd(train_cmd,
                                 '%s/log_%s.txt' % (test_dir, worker_name))
  if num_evaluator > 0:
    tf_config['task'] = {'type': 'evaluator', 'index': 0}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    set_gpu_id('')
    procs['evaluator'] = run_cmd(train_cmd,
                                 '%s/log_%s.txt' % (test_dir, 'evaluator'))

  return procs


def _ps_worker_distribute_eval(pipeline_config_path,
                               checkpoint_path,
                               test_dir,
                               num_worker,
                               num_evaluator=0):
  gpus = get_available_gpus()
  # not enough gpus, run on cpu only
  if len(gpus) < num_worker:
    gpus = [None] * num_worker
  ports = _get_ports(num_worker + 1)
  chief_or_master = 'master' if num_evaluator == 0 else 'chief'
  cluster = {
      chief_or_master: ['localhost:%d' % ports[0]],
      'worker': ['localhost:%d' % ports[i] for i in range(1, num_worker)],
      'ps': ['localhost:%d' % ports[-1]]
  }
  tf_config = {'cluster': cluster}
  procs = {}
  tf_config['task'] = {'type': chief_or_master, 'index': 0}
  os.environ['TF_CONFIG'] = json.dumps(tf_config)
  os.environ[constant.SORT_COL_BY_NAME] = '1'
  set_gpu_id(gpus[0])
  train_cmd = 'python -m easy_rec.python.eval --pipeline_config_path {} --checkpoint_path {}  \
    --distribute_eval True --eval_result_path distribute_eval_result.txt'.format(
      pipeline_config_path, checkpoint_path)
  procs[chief_or_master] = run_cmd(
      train_cmd, '%s/distribute_eval_log_%s.txt' % (test_dir, chief_or_master))
  tf_config['task'] = {'type': 'ps', 'index': 0}
  os.environ['TF_CONFIG'] = json.dumps(tf_config)
  set_gpu_id('')
  procs['ps'] = run_cmd(train_cmd,
                        '%s/distribute_eval_log_%s.txt' % (test_dir, 'ps'))

  for idx in range(num_worker - 1):
    tf_config['task'] = {'type': 'worker', 'index': idx}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    set_gpu_id(gpus[idx + 1])
    worker_name = 'worker_%d' % idx
    procs[worker_name] = run_cmd(
        train_cmd, '%s/distribute_eval_log_%s.txt' % (test_dir, worker_name))
  if num_evaluator > 0:
    tf_config['task'] = {'type': 'evaluator', 'index': 0}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    set_gpu_id('')
    procs['evaluator'] = run_cmd(
        train_cmd, '%s/distribute_eval_log_%s.txt' % (test_dir, 'evaluator'))

  return procs


def _multi_worker_mirror_train(pipeline_config_path, test_dir, num_worker):
  gpus = get_available_gpus()
  # not enough gpus, run on cpu only
  if len(gpus) < num_worker:
    gpus = [None] * num_worker
  ports = _get_ports(num_worker)
  tf_config = {
      'cluster': {
          'worker': ['localhost:%d' % ports[i] for i in range(num_worker)]
      }
  }
  procs = {}
  train_cmd = 'python -m easy_rec.python.train_eval --pipeline_config_path %s' % pipeline_config_path
  for idx in range(num_worker):
    tf_config['task'] = {'type': 'worker', 'index': idx}
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    set_gpu_id(gpus[idx])
    worker_name = 'worker_%d' % idx
    procs[worker_name] = run_cmd(train_cmd,
                                 '%s/log_%s.txt' % (test_dir, worker_name))
  return procs


def _multi_worker_hvd_train(pipeline_config_path, test_dir, num_worker):
  gpus = get_available_gpus()
  # not enough gpus, run on cpu only
  if len(gpus) < num_worker:
    gpus = ''
  else:
    gpus = ','.join(gpus)
  set_gpu_id(gpus)
  ports = _get_ports(num_worker)
  hosts = ','.join(['localhost:%d' % ports[i] for i in range(num_worker)])
  train_cmd = 'horovodrun -np %d --hosts %s python -m easy_rec.python.train_eval --pipeline_config_path %s' % (
      num_worker, hosts, pipeline_config_path)
  proc = run_cmd(train_cmd, '%s/log_hvd.txt' % test_dir)
  proc_wait(proc, timeout=1200)
  return proc.returncode == 0


def test_distributed_train_eval(pipeline_config_path,
                                test_dir,
                                total_steps=50,
                                num_evaluator=0,
                                edit_config_json=None,
                                use_hvd=False,
                                fit_on_eval=False,
                                num_epoch=0):
  logging.info('testing pipeline config %s' % pipeline_config_path)
  pipeline_config = _load_config_for_test(pipeline_config_path, test_dir,
                                          total_steps, num_epoch)
  if edit_config_json is not None:
    config_util.edit_config(pipeline_config, edit_config_json)

  if use_hvd:
    pipeline_config.train_config.sync_replicas = False
    if pipeline_config.train_config.train_distribute not in [
        DistributionStrategy.EmbeddingParallelStrategy,
        DistributionStrategy.SokStrategy
    ]:
      pipeline_config.train_config.train_distribute =\
          DistributionStrategy.HorovodStrategy

  train_config = pipeline_config.train_config
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')

  task_failed = None
  procs = None
  try:
    if use_hvd:
      return _multi_worker_hvd_train(test_pipeline_config_path, test_dir, 2)
    if train_config.train_distribute == DistributionStrategy.NoStrategy:
      num_worker = 2
      procs = _ps_worker_train(
          test_pipeline_config_path,
          test_dir,
          num_worker,
          num_evaluator,
          fit_on_eval,
          fit_on_eval_steps=int(total_steps // 2))
    elif train_config.train_distribute == DistributionStrategy.MultiWorkerMirroredStrategy:
      num_worker = 2
      procs = _multi_worker_mirror_train(test_pipeline_config_path, test_dir,
                                         num_worker)
    else:
      raise NotImplementedError

    # print proc info
    assert len(procs) > 0, 'processes are empty'
    for k, proc in procs.items():
      logging.info('%s pid: %d' % (k, proc.pid))
    task_finish_cnt = 0
    task_has_finished = {k: False for k in procs.keys()}
    while True:
      for k, proc in procs.items():
        if proc.poll() is None:
          if task_failed is not None:
            logging.error('task %s failed, %s quit' % (task_failed, k))
            proc.terminate()
            if k != 'ps':
              task_has_finished[k] = True
              task_finish_cnt += 1
            logging.info('task_finish_cnt %d' % task_finish_cnt)
        else:
          if not task_has_finished[k]:
            # process quit by itself
            if k != 'ps':
              task_finish_cnt += 1
              task_has_finished[k] = True
            logging.info('task_finish_cnt %d' % task_finish_cnt)
            if proc.returncode != 0:
              logging.error('%s failed' % k)
              task_failed = k
            else:
              logging.info('%s run successfuly' % k)

      if task_finish_cnt >= num_worker:
        break
      time.sleep(1)

  except Exception as e:
    logging.error('Exception: ' + str(e))
    raise e
  finally:
    if procs is not None:
      for k, proc in procs.items():
        if proc.poll() is None:
          logging.info('terminate %s' % k)
          proc.terminate()
    if task_failed is not None:
      logging.error('train %s failed' % pipeline_config_path)

  return task_failed is None


def test_distribute_eval_test(cur_eval_path, test_dir):
  single_work_eval_path = os.path.join(cur_eval_path, 'eval_result.txt')
  distribute_eval_path = os.path.join(test_dir, 'distribute_eval_result.txt')
  if not os.path.exists(distribute_eval_path):
    return False
  single_data = read_data_from_json_path(single_work_eval_path)
  distribute_data = read_data_from_json_path(distribute_eval_path)
  single_ret = {
      k: single_data[k]
      for k in single_data.keys()
      if 'loss' not in k and 'step' not in k
  }
  distribute_ret = {
      k: distribute_data[k] for k in distribute_data.keys() if 'loss' not in k
  }
  difference_num = 0.00001
  for k in single_ret.keys():
    if (abs(single_ret[k] - distribute_ret[k]) > difference_num):
      logging.error(
          'distribute_eval difference[%.8f] large than threshold[%.8f]' %
          (abs(single_ret[k] - distribute_ret[k]), difference_num))
      return False
  return True


def test_distributed_eval(pipeline_config_path,
                          checkpoint_path,
                          test_dir,
                          total_steps=50,
                          num_evaluator=0):
  logging.info('testing pipeline config %s' % pipeline_config_path)
  pipeline_config = _load_config_for_distribute_eval(pipeline_config_path,
                                                     test_dir)
  train_config = pipeline_config.train_config
  config_util.save_pipeline_config(pipeline_config, test_dir)
  test_pipeline_config_path = os.path.join(test_dir, 'pipeline.config')

  task_failed = None
  procs = None
  is_equal = False
  try:
    if train_config.train_distribute == DistributionStrategy.NoStrategy:
      num_worker = 2
      procs = _ps_worker_distribute_eval(test_pipeline_config_path,
                                         checkpoint_path, test_dir, num_worker,
                                         num_evaluator)
    else:
      raise NotImplementedError

    # print proc info
    assert len(procs) > 0, 'processes are empty'
    for k, proc in procs.items():
      logging.info('%s pid: %d' % (k, proc.pid))
    task_finish_cnt = 0
    task_has_finished = {k: False for k in procs.keys()}
    while True:
      for k, proc in procs.items():
        if proc.poll() is None:
          if task_failed is not None:
            logging.error('task %s failed, %s quit' % (task_failed, k))
            proc.terminate()
            if k != 'ps':
              task_has_finished[k] = True
              task_finish_cnt += 1
            logging.info('task_finish_cnt %d' % task_finish_cnt)
        else:
          if not task_has_finished[k]:
            # process quit by itself
            if k != 'ps':
              task_finish_cnt += 1
              task_has_finished[k] = True
            logging.info('task_finish_cnt %d' % task_finish_cnt)
            if proc.returncode != 0:
              logging.error('%s failed' % k)
              task_failed = k
            else:
              logging.info('%s run successfuly' % k)
      if task_finish_cnt >= num_worker:
        break
      time.sleep(1)

    is_equal = test_distribute_eval_test(checkpoint_path, test_dir)

  except Exception as e:
    logging.error('Exception: ' + str(e))
    raise e
  finally:
    if procs is not None:
      for k, proc in procs.items():
        if proc.poll() is None:
          logging.info('terminate %s' % k)
          proc.terminate()
    if task_failed is not None:
      logging.error('eval %s failed[%s]' % (pipeline_config_path, task_failed))

  eval_success = (task_failed is None) and is_equal
  return eval_success
