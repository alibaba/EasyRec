# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import sys

import tensorflow as tf

from easy_rec.python.input.input import Input

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class KafkaInput(Input):

  def __init__(self,
               data_config,
               feature_config,
               kafka_config,
               task_index=0,
               task_num=1):
    super(KafkaInput, self).__init__(data_config, feature_config, '',
                                     task_index, task_num)
    self._kafka = kafka_config

  def _parse_csv(self, line):
    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    def _check_data(line):
      sep = self._data_config.separator
      if type(sep) != type(str):
        sep = sep.encode('utf-8')
      field_num = len(line[0].split(sep))
      assert field_num == len(record_defaults),\
          'sep[%s] maybe invalid: field_num=%d, required_num=%d' % (sep, field_num, len(record_defaults))
      return True

    check_op = tf.py_func(_check_data, [line], Tout=tf.bool)
    with tf.control_dependencies([check_op]):
      fields = tf.decode_csv(
          line,
          field_delim=self._data_config.separator,
          record_defaults=record_defaults,
          name='decode_csv')

    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}

    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _build(self, mode, params):
    try:
      import tensorflow_io.kafka as kafka_io
    except ImportError:
      logging.error(
          'Please install tensorflow-io, '
          'version compatibility can refer to https://github.com/tensorflow/io#tensorflow-version-compatibility'
      )

    num_parallel_calls = self._data_config.num_parallel_calls
    if mode == tf.estimator.ModeKeys.TRAIN:
      train = self._kafka
      topics = []
      i = self._task_index
      assert len(train.offset) == 1 or len(train.offset) == train.partitions, \
          'number of train.offset must be 1 or train.partitions'
      while i < train.partitions:
        offset_i = train.offset[i] if i < len(
            train.offset) else train.offset[-1]
        topics.append(train.topic + ':' + str(i) + ':' + str(offset_i) + ':-1')
        i = i + self._task_num

      logging.info(
          'train kafka server: %s topic: %s task_num: %d task_index: %d topics: %s'
          %
          (train.server, train.topic, self._task_num, self._task_index, topics))
      if len(topics) == 0:
        logging.info('train kafka topic is empty')
        sys.exit(1)

      dataset = kafka_io.KafkaDataset(
          topics, servers=train.server, group=train.group, eof=False)
      dataset = dataset.repeat(1)
    else:
      eval = self._kafka
      topics = []
      i = 0
      assert len(eval.offset) == 1 or len(eval.offset) == eval.partitions, \
          'number of eval.offset must be 1 or eval.partitions'
      while i < eval.partitions:
        offset_i = eval.offset[i] if i < len(eval.offset) else eval.offset[-1]
        topics.append(eval.topic + ':' + str(i) + ':' + str(eval.offset) +
                      ':-1')
        i = i + 1

      logging.info(
          'eval kafka server: %s topic: %s task_num: %d task_index: %d topics: %s'
          % (eval.server, eval.topic, self._task_num, self._task_index, topics))

      if len(topics) == 0:
        logging.info('eval kafka topic is empty')
        sys.exit(1)

      dataset = kafka_io.KafkaDataset(
          topics, servers=eval.server, group=eval.group, eof=False)
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self._data_config.batch_size)
    dataset = dataset.map(
        self._parse_csv, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    dataset = dataset.map(
        map_func=self._preprocess, num_parallel_calls=num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
