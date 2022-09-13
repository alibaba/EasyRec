# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kafka Dataset."""

import logging
import traceback

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

try:
  from easy_rec.python.ops import gen_kafka_ops
except ImportError:
  logging.warning('failed to import gen_kafka_ops: %s' % traceback.format_exc())


class KafkaDataset(dataset_ops.Dataset):
  """A Kafka Dataset that consumes the message."""

  def __init__(self,
               topics,
               servers='localhost',
               group='',
               eof=False,
               timeout=1000,
               config_global=None,
               config_topic=None,
               message_key=False,
               message_offset=False):
    """Create a KafkaReader.

    Args:
      topics: A `tf.string` tensor containing one or more subscriptions,
              in the format of [topic:partition:offset:length],
              by default length is -1 for unlimited.
      servers: A list of bootstrap servers.
      group: The consumer group id.
      eof: If True, the kafka reader will stop on EOF.
      timeout: The timeout value for the Kafka Consumer to wait
               (in millisecond).
      config_global: A `tf.string` tensor containing global configuration
                     properties in [Key=Value] format,
                     eg. ["enable.auto.commit=false",
                          "heartbeat.interval.ms=2000"],
                     please refer to 'Global configuration properties'
                     in librdkafka doc.
      config_topic: A `tf.string` tensor containing topic configuration
                    properties in [Key=Value] format,
                    eg. ["auto.offset.reset=earliest"],
                    please refer to 'Topic configuration properties'
                    in librdkafka doc.
      message_key: If True, the kafka will output both message value and key.
      message_offset: If True, the kafka will output both message value and offset.
    """
    self._topics = ops.convert_to_tensor(
        topics, dtype=dtypes.string, name='topics')
    self._servers = ops.convert_to_tensor(
        servers, dtype=dtypes.string, name='servers')
    self._group = ops.convert_to_tensor(
        group, dtype=dtypes.string, name='group')
    self._eof = ops.convert_to_tensor(eof, dtype=dtypes.bool, name='eof')
    self._timeout = ops.convert_to_tensor(
        timeout, dtype=dtypes.int64, name='timeout')
    config_global = config_global if config_global else []
    self._config_global = ops.convert_to_tensor(
        config_global, dtype=dtypes.string, name='config_global')
    config_topic = config_topic if config_topic else []
    self._config_topic = ops.convert_to_tensor(
        config_topic, dtype=dtypes.string, name='config_topic')
    self._message_key = message_key
    self._message_offset = message_offset
    super(KafkaDataset, self).__init__()

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return gen_kafka_ops.io_kafka_dataset_v2(
        self._topics,
        self._servers,
        self._group,
        self._eof,
        self._timeout,
        self._config_global,
        self._config_topic,
        self._message_key,
        self._message_offset,
    )

  @property
  def output_classes(self):
    if self._message_key ^ self._message_offset:
      return (ops.Tensor, ops.Tensor)
    elif self._message_key and self._message_offset:
      return (ops.Tensor, ops.Tensor, ops.Tensor)
    return (ops.Tensor)

  @property
  def output_shapes(self):
    if self._message_key ^ self._message_offset:
      return ((tensor_shape.TensorShape([]), tensor_shape.TensorShape([])))
    elif self._message_key and self._message_offset:
      return ((tensor_shape.TensorShape([]), tensor_shape.TensorShape([]),
               tensor_shape.TensorShape([])))
    return ((tensor_shape.TensorShape([])))

  @property
  def output_types(self):
    if self._message_key ^ self._message_offset:
      return ((dtypes.string, dtypes.string))
    elif self._message_key and self._message_offset:
      return ((dtypes.string, dtypes.string, dtypes.string))
    return ((dtypes.string))


def write_kafka_v2(message, topic, servers='localhost', name=None):
  """Write kafka.

  Args:
    message: A `Tensor` of type `string`. 0-D.
    topic: A `tf.string` tensor containing one subscription,
      in the format of topic:partition.
    servers: A list of bootstrap servers.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. 0-D.
  """
  return gen_kafka_ops.io_write_kafka_v2(
      message=message, topic=topic, servers=servers, name=name)
