# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import os
import threading
import time
import traceback
import unittest

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.platform import gfile

from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.input.kafka_dataset import KafkaDataset
from easy_rec.python.utils import numpy_utils
from easy_rec.python.utils import test_utils

try:
  import kafka
  from kafka import KafkaProducer, KafkaAdminClient
  from kafka.admin import NewTopic
except ImportError:
  logging.warning('kafka-python is not installed: %s' % traceback.format_exc())


class KafkaTest(tf.test.TestCase):

  def setUp(self):
    self._success = True
    self._test_dir = test_utils.get_tmp_dir()
    if self._testMethodName == 'test_session':
      self._kafka_server_proc = None
      self._zookeeper_proc = None
      return

    logging.info('Testing %s.%s, test_dir=%s' %
                 (type(self).__name__, self._testMethodName, self._test_dir))
    self._log_dir = os.path.join(self._test_dir, 'logs')
    if not gfile.IsDirectory(self._log_dir):
      gfile.MakeDirs(self._log_dir)

    self._kafka_servers = ['127.0.0.1:9092']
    self._test_topic = 'kafka_op_test_topic'

    if 'kafka_install_dir' in os.environ:
      kafka_install_dir = os.environ.get('kafka_install_dir', None)

      zookeeper_config_raw = '%s/config/zookeeper.properties' % kafka_install_dir
      zookeeper_config = os.path.join(self._test_dir, 'zookeeper.properties')
      with open(zookeeper_config, 'w') as fout:
        with open(zookeeper_config_raw, 'r') as fin:
          for line_str in fin:
            if line_str.startswith('dataDir='):
              fout.write('dataDir=%s/zookeeper\n' % self._test_dir)
            else:
              fout.write(line_str)
      cmd = 'bash %s/bin/zookeeper-server-start.sh %s' % (kafka_install_dir,
                                                          zookeeper_config)
      log_file = os.path.join(self._log_dir, 'zookeeper.log')
      self._zookeeper_proc = test_utils.run_cmd(cmd, log_file)

      kafka_config_raw = '%s/config/server.properties' % kafka_install_dir
      kafka_config = os.path.join(self._test_dir, 'server.properties')
      with open(kafka_config, 'w') as fout:
        with open(kafka_config_raw, 'r') as fin:
          for line_str in fin:
            if line_str.startswith('log.dirs='):
              fout.write('log.dirs=%s/kafka\n' % self._test_dir)
            else:
              fout.write(line_str)
      cmd = 'bash %s/bin/kafka-server-start.sh %s' % (kafka_install_dir,
                                                      kafka_config)
      log_file = os.path.join(self._log_dir, 'kafka_server.log')
      self._kafka_server_proc = test_utils.run_cmd(cmd, log_file)

      started = False
      while not started:
        if self._kafka_server_proc.poll(
        ) and self._kafka_server_proc.returncode:
          logging.warning('start kafka server failed, will retry.')
          os.system('cat %s' % log_file)
          self._kafka_server_proc = test_utils.run_cmd(cmd, log_file)
          time.sleep(5)
        else:
          try:
            admin_clt = KafkaAdminClient(bootstrap_servers=self._kafka_servers)
            logging.info('old topics: %s' % (','.join(admin_clt.list_topics())))
            admin_clt.close()
            started = True
          except kafka.errors.NoBrokersAvailable:
            time.sleep(2)
      self._create_topic()
    else:
      self._zookeeper_proc = None
      self._kafka_server_proc = None
    self._should_stop = False
    self._producer = None

  def _create_topic(self, num_partitions=2):
    admin_clt = KafkaAdminClient(bootstrap_servers=self._kafka_servers)

    logging.info('create topic: %s' % self._test_topic)
    topic_list = [
        NewTopic(
            name=self._test_topic,
            num_partitions=num_partitions,
            replication_factor=1)
    ]

    admin_clt.create_topics(new_topics=topic_list, validate_only=False)
    logging.info('all topics: %s' % (','.join(admin_clt.list_topics())))
    admin_clt.close()

  def _create_producer(self, generate_func):
    # start produce thread

    prod = threading.Thread(target=generate_func)
    prod.start()
    return prod

  def _stop_producer(self):
    if self._producer is not None:
      self._should_stop = True
      self._producer.join()

  def tearDown(self):
    try:
      self._stop_producer()
      if self._kafka_server_proc is not None:
        self._kafka_server_proc.terminate()
    except Exception as ex:
      logging.warning('exception terminate kafka proc: %s' % str(ex))

    try:
      if self._zookeeper_proc is not None:
        self._zookeeper_proc.terminate()
    except Exception as ex:
      logging.warning('exception terminate zookeeper proc: %s' % str(ex))

    test_utils.set_gpu_id(None)
    if self._success:
      test_utils.clean_up(self._test_dir)

  @unittest.skipIf('kafka_install_dir' not in os.environ,
                   'Only execute when kafka is available')
  def test_kafka_ops(self):
    try:
      test_utils.set_gpu_id(None)

      def _generate():
        producer = KafkaProducer(
            bootstrap_servers=self._kafka_servers, api_version=(0, 10, 1))
        i = 0
        while not self._should_stop:
          msg = 'user_id_%d' % i
          producer.send(self._test_topic, msg)
        producer.close()

      self._producer = self._create_producer(_generate)

      group = 'dataset_consumer'
      k = KafkaDataset(
          servers=self._kafka_servers[0],
          topics=[self._test_topic + ':0', self._test_topic + ':1'],
          group=group,
          eof=True,
          # control the maximal read of each partition
          config_global=['max.partition.fetch.bytes=1048576'],
          message_key=True,
          message_offset=True)

      batch_dataset = k.batch(5)

      iterator = iterator_ops.Iterator.from_structure(
          batch_dataset.output_types)
      init_batch_op = iterator.make_initializer(batch_dataset)
      get_next = iterator.get_next()

      sess = tf.Session()
      sess.run(init_batch_op)

      p = sess.run(get_next)

      self.assertEquals(len(p), 3)
      offset = p[2]
      self.assertEquals(offset[0], '0:0')
      self.assertEquals(offset[1], '0:1')

      p = sess.run(get_next)
      offset = p[2]
      self.assertEquals(offset[0], '0:5')
      self.assertEquals(offset[1], '0:6')

      max_iter = 300
      while max_iter > 0:
        sess.run(get_next)
        max_iter -= 1
    except tf.errors.OutOfRangeError:
      pass
    except Exception as ex:
      self._success = False
      raise ex

  @unittest.skipIf('kafka_install_dir' not in os.environ,
                   'Only execute when kafka is available')
  def test_kafka_train(self):
    try:
      # start produce thread
      self._producer = self._create_producer(self._generate)

      test_utils.set_gpu_id(None)

      self._success = test_utils.test_single_train_eval(
          'samples/model_config/deepfm_combo_avazu_kafka.config',
          self._test_dir)
      self.assertTrue(self._success)
    except Exception as ex:
      self._success = False
      raise ex

  def _generate(self):
    producer = KafkaProducer(
        bootstrap_servers=self._kafka_servers, api_version=(0, 10, 1))
    while not self._should_stop:
      with open('data/test/dwd_avazu_ctr_deepmodel_10w.csv', 'r') as fin:
        for line_str in fin:
          line_str = line_str.strip()
          if self._should_stop:
            break
          if six.PY3:
            line_str = line_str.encode('utf-8')
          producer.send(self._test_topic, line_str)
    producer.close()
    logging.info('data generation thread done.')

  @unittest.skipIf('kafka_install_dir' not in os.environ,
                   'Only execute when kafka is available')
  def test_kafka_train_chief_redundant(self):
    try:
      # start produce thread
      self._producer = self._create_producer(self._generate)

      test_utils.set_gpu_id(None)

      self._success = test_utils.test_distributed_train_eval(
          'samples/model_config/deepfm_combo_avazu_kafka_chief_redundant.config',
          self._test_dir,
          num_evaluator=1)
      self.assertTrue(self._success)
    except Exception as ex:
      self._success = False
      raise ex

  @unittest.skipIf('kafka_install_dir' not in os.environ,
                   'Only execute when kafka is available')
  def test_kafka_train_v2(self):
    try:
      # start produce thread
      self._producer = self._create_producer(self._generate)

      test_utils.set_gpu_id(None)

      self._success = test_utils.test_single_train_eval(
          'samples/model_config/deepfm_combo_avazu_kafka_time_offset.config',
          self._test_dir)

      self.assertTrue(self._success)
    except Exception as ex:
      self._success = False
      raise ex

  @unittest.skipIf(
      'kafka_install_dir' not in os.environ or 'oss_path' not in os.environ or
      'oss_endpoint' not in os.environ and 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ, 'Only execute when kafka is available')
  def test_kafka_processor(self):
    self._test_kafka_processor(
        'samples/model_config/taobao_fg_incr_save.config')

  @unittest.skipIf(
      'kafka_install_dir' not in os.environ or 'oss_path' not in os.environ or
      'oss_endpoint' not in os.environ and 'oss_ak' not in os.environ or
      'oss_sk' not in os.environ, 'Only execute when kafka is available')
  def test_kafka_processor_ev(self):
    self._test_kafka_processor(
        'samples/model_config/taobao_fg_incr_save_ev.config')

  def _test_kafka_processor(self, config_path):
    self._success = False
    success = test_utils.test_distributed_train_eval(
        config_path, self._test_dir, total_steps=500)
    self.assertTrue(success)
    export_cmd = """
       python -m easy_rec.python.export --pipeline_config_path %s/pipeline.config
           --export_dir %s/export/sep/ --oss_path=%s --oss_ak=%s --oss_sk=%s --oss_endpoint=%s
           --asset_files ./samples/rtp_fg/fg.json
           --checkpoint_path %s/train/model.ckpt-0
    """ % (self._test_dir, self._test_dir, os.environ['oss_path'],
           os.environ['oss_ak'], os.environ['oss_sk'],
           os.environ['oss_endpoint'], self._test_dir)
    proc = test_utils.run_cmd(export_cmd,
                              '%s/log_export_sep.txt' % self._test_dir)
    proc.wait()
    self.assertTrue(proc.returncode == 0)
    files = gfile.Glob(os.path.join(self._test_dir, 'export/sep/[1-9][0-9]*'))
    export_sep_dir = files[0]

    predict_cmd = """
        python -m easy_rec.python.inference.processor.test --saved_model_dir %s
           --input_path data/test/rtp/taobao_test_feature.txt
           --output_path %s/processor.out  --test_dir %s
     """ % (export_sep_dir, self._test_dir, self._test_dir)
    envs = dict(os.environ)
    envs['PROCESSOR_TEST'] = '1'
    proc = test_utils.run_cmd(
        predict_cmd, '%s/log_processor.txt' % self._test_dir, env=envs)
    proc.wait()
    self.assertTrue(proc.returncode == 0)

    with open('%s/processor.out' % self._test_dir, 'r') as fin:
      processor_out = []
      for line_str in fin:
        line_str = line_str.strip()
        processor_out.append(json.loads(line_str))

    predictor = Predictor(os.path.join(self._test_dir, 'train/export/final/'))
    with open('data/test/rtp/taobao_test_feature.txt', 'r') as fin:
      inputs = []
      for line_str in fin:
        line_str = line_str.strip()
        line_tok = line_str.split(';')[-1]
        line_tok = line_tok.split(chr(2))
        inputs.append(line_tok)
    output_res = predictor.predict(inputs, batch_size=1024)

    with open('%s/predictor.out' % self._test_dir, 'w') as fout:
      for i in range(len(output_res)):
        fout.write(
            json.dumps(output_res[i], cls=numpy_utils.NumpyEncoder) + '\n')

    for i in range(len(output_res)):
      val0 = output_res[i]['probs']
      val1 = processor_out[i]['probs']
      diff = np.abs(val0 - val1)
      assert diff < 1e-4, 'too much difference[%.6f] >= 1e-4' % diff
    self._success = True

  @unittest.skipIf('kafka_install_dir' not in os.environ,
                   'Only execute when kafka is available')
  def test_kafka_train_v3(self):
    try:
      # start produce thread
      self._producer = self._create_producer(self._generate)

      test_utils.set_gpu_id(None)

      self._success = test_utils.test_single_train_eval(
          'samples/model_config/deepfm_combo_avazu_kafka_time_offset2.config',
          self._test_dir)

      self.assertTrue(self._success)
    except Exception as ex:
      self._success = False
      raise ex


if __name__ == '__main__':
  tf.test.main()
