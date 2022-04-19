# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import logging
import argparse
from kafka import KafkaProducer
from kafka import KafkaConsumer, KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.structs import TopicPartition

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--servers', type=str, default='localhost:9092')
  parser.add_argument('--topic', type=str, default=None)
  parser.add_argument('--group', type=str, default='consumer')
  parser.add_argument('--partitions', type=str, default=None)
  parser.add_argument('--timeout', type=float, default=float('inf'))
  args = parser.parse_args()

  if args.topic is None:
    logging.error('--topic is not set')
    sys.exit(1)
 
  servers = args.servers.split(',')
  consumer = KafkaConsumer(group_id=args.group, bootstrap_servers=servers,
      consumer_timeout_ms=args.timeout * 1000)

  if args.partitions is not None:
    partitions = [ int(x) for x in args.partitions.split(',') ]
  else:
    partitions = consumer.partitions_for_topic(args.topic)
  logging.info('partitions: %s' % partitions)

  topics = [ TopicPartition(topic=args.topic, partition=part_id) \
             for part_id in partitions ]
  consumer.assign(topics)
  consumer.seek_to_beginning()
  
  record_id = 0
  for x in consumer:
    logging.info("%d: key=%s\toffset=%d\ttimestamp=%d\tlen=%d" % (record_id, x.key, x.offset,
        x.timestamp, len(x.value)))
    record_id += 1
