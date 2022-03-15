# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import tensorflow as tf
from kafka import KafkaProducer
from tensorflow.python.data.ops import iterator_ops

from easy_rec.python.utils.kafka_utils import kafka_maker


class KafkaTest(tf.test.TestCase):

  def setUp(self):
    pass

  # @unittest.skip('Only execute when kafka is available')
  def test_kafka_ops(self):
    servers = ['localhost:9092']
    topic = 'kafka-test-topic'
    producer = KafkaProducer(
        bootstrap_servers=servers, api_version=(0, 10, 1))
    for i in range(30, 40, 1):
      msg = 'user_id_%d' % i
      producer.send(topic, msg)
    producer.close()

    # FIXME topics="topic:partition:offset:length"
    kafka_dataset = kafka_maker()
    group = 'dataset_consumer'
    k = kafka_dataset(
        servers=servers[0],
        topics=[topic],
        group=group,
        eof=False,
        config_topic='auto.offset.reset=largest',
        message_key=True,
        message_offset=True)

    batch_dataset = k.batch(2)

    iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
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
    self.assertEquals(offset[0], '0:2')
    self.assertEquals(offset[1], '0:3')


if __name__ == '__main__':
  tf.test.main()
