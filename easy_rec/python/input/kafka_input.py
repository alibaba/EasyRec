# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import sys

import tensorflow as tf

from easy_rec.python.input.input import Input

try:
  from tensorflow_io.kafka import KafkaDataset
except ImportError:
  from easy_rec.python.input.kafka_dataset import KafkaDatasetV2 as KafkaDataset
except NotImplementedError:
  from easy_rec.python.input.kafka_dataset import KafkaDatasetV2 as KafkaDataset
except ValueError:
  from easy_rec.python.input.kafka_dataset import KafkaDatasetV2 as KafkaDataset

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class KafkaInput(Input):

  DATA_OFFSET = 'DATA_OFFSET'

  def __init__(self,
               data_config,
               feature_config,
               kafka_config,
               task_index=0,
               task_num=1):
    super(KafkaInput, self).__init__(data_config, feature_config, '',
                                     task_index, task_num)
    self._kafka = kafka_config

  def _preprocess(self, field_dict):
    output_dict = super(KafkaInput, self)._preprocess(field_dict)
    output_dict[Input.DATA_OFFSET] = field_dict[Input.DATA_OFFSET]

    if Input.DATA_OFFSET not in self._appended_fields: 
      self._appended_fields.append(Input.DATA_OFFSET)
    return output_dict

  def _parse_csv(self, line, message_key, message_offset):
    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    fields = tf.decode_csv(
        line,
        use_quote_delim=False,
        field_delim=self._data_config.separator,
        record_defaults=record_defaults,
        name='decode_csv')

    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}

    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]

    offset_vals = tf.strings.split(message_offset, ':').values
    offset_vals = tf.reshape(offset_vals, [-1, 2])
    offset_vals = offset_vals[:, 1]

    inputs[Input.DATA_OFFSET] = tf.string_to_number(offset_vals, tf.int64)
    return inputs

  def _build(self, mode, params):
    num_parallel_calls = self._data_config.num_parallel_calls
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_kafka = self._kafka
      topics = []
      assert train_kafka.partitions == len(train_kafka.offset)
      for part_id, offset in enumerate(train_kafka.offset):
        if (part_id % self._task_num) == self._task_index:
          topics.append('%s:%d:%d:-1' % (train_kafka.topic, self._task_index, offset))
      assert len(topics) > 0, 'no partitions are assigned for train task(%d)' % self._task_index

      logging.info(
          'train kafka server: %s topic: %s task_num: %d task_index: %d topics: %s'
          %
          (train_kafka.server, train_kafka.topic, self._task_num, self._task_index, topics))

      dataset = KafkaDataset(
          topics,
          servers=train_kafka.server,
          group=train_kafka.group,
          eof=False,
          message_key=True,
          message_offset=True)
      dataset = dataset.repeat(1)
    else:
      eval_kafka = self._kafka
      topics = [
          '%s:%d:%d:-1' % (eval_kafka.topic, part_id, eval_kafka.offset[part_id])
          for part_id in range(eval_kafka.partitions)
      ]

      logging.info(
          'eval kafka server: %s topic: %s task_num: %d task_index: %d topics: %s'
          % (eval_kafka.server, eval_kafka.topic, self._task_num, self._task_index, topics))

      assert len(topics) > 0, 'eval kafka topic is not set'

      dataset = tf.data.Dataset.from_tensor_slices(topics)\
          .interleave(lambda x: KafkaDataset(
              x, servers=eval_kafka.server, group=eval_kafka.group, eof=True,
              config_topic='auto.offset.reset=largest',
              message_key=True, message_offset=True).repeat(1),
              cycle_length=len(topics), block_length=1)

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
