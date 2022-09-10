# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging
import sys

# from kafka import KafkaConsumer
from kafka import KafkaAdminClient
from kafka import KafkaProducer
from kafka.admin import NewTopic

# from kafka.structs import TopicPartition

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--servers', type=str, default='localhost:9092')
  parser.add_argument('--topic', type=str, default=None)
  parser.add_argument('--group', type=str, default='consumer')
  parser.add_argument('--partitions', type=str, default=None)
  parser.add_argument('--timeout', type=float, default=float('inf'))
  # file to send
  parser.add_argument('--input_path', type=str, default=None)
  args = parser.parse_args()

  if args.input_path is None:
    logging.error('input_path is not set')
    sys.exit(1)

  if args.topic is None:
    logging.error('topic is not set')
    sys.exit(1)

  servers = args.servers.split(',')

  admin_clt = KafkaAdminClient(bootstrap_servers=servers)
  if args.topic not in admin_clt.list_topics():
    admin_clt.create_topics(
        new_topics=[
            NewTopic(
                name=args.topic,
                num_partitions=1,
                replication_factor=1,
                topic_configs={'max.message.bytes': 1024 * 1024 * 1024})
        ],
        validate_only=False)
    logging.info('create increment save topic: %s' % args.topic)
  admin_clt.close()

  producer = KafkaProducer(
      bootstrap_servers=servers,
      request_timeout_ms=args.timeout * 1000,
      api_version=(0, 10, 1))

  i = 1
  with open(args.input_path, 'r') as fin:
    for line_str in fin:
      producer.send(args.topic, line_str.encode('utf-8'))
      i += 1
      break
      if i % 100 == 0:
        logging.info('progress: %d' % i)
  producer.close()
