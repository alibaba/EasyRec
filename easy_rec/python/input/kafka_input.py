# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import sys
import traceback
import json
import six

import tensorflow as tf

from easy_rec.python.input.input import Input
from easy_rec.python.input.kafka_dataset import KafkaDataset

try:
  from kafka import KafkaConsumer, TopicPartition
except ImportError as ex:
  logging.warning('kafka-python is not installed: %s' % traceback.format_exc(ex))

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
    self._offset_dict = {}
    if self._kafka is not None:
      consumer = KafkaConsumer(group_id='kafka_dataset_consumer', 
           bootstrap_servers=[self._kafka.server],
           api_version_auto_timeout_ms=10000) # in miliseconds
      partitions = consumer.partitions_for_topic(self._kafka.topic)
      num_partition = len(partitions)
      logging.info('all partitions[%d]: %s' % (num_partition, partitions))

      # each topic in the format: 
      #     topic:partition_id:offset
      self._topics = []

      # determine kafka offsets for each partition
      if self._kafka.offset_info:
        offset_dict = json.loads(self._kafka.offset_info)
        if 'timestamp' in self._kafka.offset_info:
          timestamp = offset_dict['timestamp'] 
          input_map = { TopicPartition(partition=part_id, topic=self._kafka.topic) : timestamp * 1000 \
              for part_id in partitions }
          part_offsets = consumer.offsets_for_times(input_map)
          # {TopicPartition(topic=u'kafka_data_20220408', partition=0):
          #    OffsetAndTimestamp(offset=2, timestamp=1650611437895)}
          for part in part_offsets:
            self._offset_dict[part.partition] = part_offsets[part].offset
            logging.info('find offset by time, topic[%s], partition[%d], timestamp[%ss], offset[%d], offset_timestamp[%dms]' % \
                (self._kafka.topic, part.partition, timestamp, part_offsets[part].offset, part_offsets[part].timestamp))
        else:
          for part in offset_dict:
            part_id = int(part)
            if (part_id % self._task_num) == self._task_index:
              self._offset_dict[part_id] = offset_dict[part]

      for part_id in range(num_partition):
        if (part_id % self._task_num) == self._task_index:
          offset = self._offset_dict.get(part_id, 0)
          self._topics.append('%s:%d:%d' % (self._kafka.topic, part_id, offset))
      logging.info('assigned topic partitions: %s' % (','.join(self._topics)))
      assert len(self._topics) > 0, 'no partitions are assigned for this task(%d/%d)' % (
         self._task_index, self._task_num)
    else:
      self._topics = None

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

    # record current offset
    def _parse_offset(message_offset):
      for kv in message_offset:
        if six.PY3:
          kv = kv.decode('utf-8')
        k,v = kv.split(':')
        v = int(v)
        if k not in self._offset_dict or v > self._offset_dict[k]:
          self._offset_dict[k] = v
      return json.dumps(self._offset_dict) 
       
    inputs[Input.DATA_OFFSET] = tf.py_func(_parse_offset, [message_offset], tf.string) 
    return inputs

  def _preprocess(self, field_dict):
    output_dict = super(KafkaInput, self)._preprocess(field_dict)

    # append offset fields
    if Input.DATA_OFFSET in field_dict:
      output_dict[Input.DATA_OFFSET] = field_dict[Input.DATA_OFFSET]

    # for _get_features to include DATA_OFFSET
    if Input.DATA_OFFSET not in self._appended_fields: 
      self._appended_fields.append(Input.DATA_OFFSET)

    return output_dict

  def _build(self, mode, params):
    num_parallel_calls = self._data_config.num_parallel_calls
    if mode == tf.estimator.ModeKeys.TRAIN:
      assert self._kafka is not None, "kafka_train_input is not set."
      train_kafka = self._kafka
      logging.info(
          'train kafka server: %s topic: %s task_num: %d task_index: %d topics: %s'
          %
          (train_kafka.server, train_kafka.topic, self._task_num, self._task_index, self._topics))

      dataset = KafkaDataset(
          self._topics,
          servers=train_kafka.server,
          group=train_kafka.group,
          eof=False,
          config_global = list(self._kafka.config_global),
          config_topic = list(self._kafka.config_topic),
          message_key=True,
          message_offset=True)
    else:
      eval_kafka = self._kafka
      assert self._kafka is not None, "kafka_eval_input is not set."
 
      logging.info(
          'eval kafka server: %s topic: %s task_num: %d task_index: %d topics: %s'
          % (eval_kafka.server, eval_kafka.topic, self._task_num, self._task_index, self._topics))

      dataset = KafkaDataset(self._topics, servers=self._kafka.server, 
              group=eval_kafka.group, eof=True,
              config_global = list(self._kafka.config_global),
              config_topic = list(self._kafka.config_topic),
              message_key=True, message_offset=True)

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
