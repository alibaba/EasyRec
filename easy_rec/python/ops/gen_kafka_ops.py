"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: kafka_ops_deprecated.cc
"""

import logging
import os
import traceback

import six as _six
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.util.tf_export import tf_export

import easy_rec

kafka_module = None
if easy_rec.ops_dir is not None:
  kafka_ops_path = os.path.join(easy_rec.ops_dir, 'kafka.so')
  if os.path.exists(kafka_ops_path):
    try:
      kafka_module = tf.load_op_library(kafka_ops_path)
    except Exception:
      logging.warning('load %s failed: %s' %
                      (kafka_ops_path, traceback.format_exc()))


@tf_export('io_kafka_dataset_v2')
def io_kafka_dataset_v2(topics,
                        servers,
                        group,
                        eof,
                        timeout,
                        config_global,
                        config_topic,
                        message_key,
                        message_offset,
                        name=None):
  """Creates a dataset that emits the messages of one or more Kafka topics.

  Args:
    topics: A `Tensor` of type `string`.
      A `tf.string` tensor containing one or more subscriptions,
      in the format of [topic:partition:offset].
    servers: A `Tensor` of type `string`. A list of bootstrap servers.
    group: A `Tensor` of type `string`. The consumer group id.
    eof: A `Tensor` of type `bool`.
      If True, the kafka reader will stop on EOF.
    timeout: A `Tensor` of type `int64`.
      The timeout value for the Kafka Consumer to wait
      (in millisecond).
    config_global: A `Tensor` of type `string`.
      A `tf.string` tensor containing global configuration
      properties in [Key=Value] format,
      eg. ["enable.auto.commit=false", "heartbeat.interval.ms=2000"],
      please refer to 'Global configuration properties' in librdkafka doc.
    config_topic: A `Tensor` of type `string`.
      A `tf.string` tensor containing topic configuration
      properties in [Key=Value] format, eg. ["auto.offset.reset=earliest"],
      please refer to 'Topic configuration properties' in librdkafka doc.
    message_key: A `Tensor` of type `bool`.
    message_offset: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  return kafka_module.io_kafka_dataset_v2(
      topics=topics,
      servers=servers,
      group=group,
      eof=eof,
      timeout=timeout,
      config_global=config_global,
      config_topic=config_topic,
      message_key=message_key,
      message_offset=message_offset,
      name=name)


def io_kafka_dataset_eager_fallback(topics,
                                    servers,
                                    group,
                                    eof,
                                    timeout,
                                    config_global,
                                    config_topic,
                                    message_key,
                                    message_offset,
                                    name=None,
                                    ctx=None):
  """This is the slowpath function for Eager mode.

  This is for function io_kafka_dataset
  """
  _ctx = ctx if ctx else _context.context()
  topics = _ops.convert_to_tensor(topics, _dtypes.string)
  servers = _ops.convert_to_tensor(servers, _dtypes.string)
  group = _ops.convert_to_tensor(group, _dtypes.string)
  eof = _ops.convert_to_tensor(eof, _dtypes.bool)
  timeout = _ops.convert_to_tensor(timeout, _dtypes.int64)
  config_global = _ops.convert_to_tensor(config_global, _dtypes.string)
  config_topic = _ops.convert_to_tensor(config_topic, _dtypes.string)
  message_key = _ops.convert_to_tensor(message_key, _dtypes.bool)
  message_offset = _ops.convert_to_tensor(message_offset, _dtypes.bool)
  _inputs_flat = [
      topics, servers, group, eof, timeout, config_global, config_topic,
      message_key, message_offset
  ]
  _attrs = None
  _result = _execute.execute(
      b'IOKafkaDataset',
      1,
      inputs=_inputs_flat,
      attrs=_attrs,
      ctx=_ctx,
      name=name)
  _execute.record_gradient('IOKafkaDataset', _inputs_flat, _attrs, _result,
                           name)
  _result, = _result
  return _result


@tf_export('io_write_kafka_v2')
def io_write_kafka_v2(message, topic, servers, name=None):
  r"""TODO: add doc.

  Args:
    message: A `Tensor` of type `string`.
    topic: A `Tensor` of type `string`.
    servers: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _op = kafka_module.io_write_kafka_v2(
        message=message, topic=topic, servers=servers, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    _execute.record_gradient('IOWriteKafka', _inputs_flat, _attrs, _result,
                             name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
          _ctx._context_handle, _ctx._eager_context.device_name, 'IOWriteKafka',
          name, _ctx._post_execution_callbacks, message, topic, servers)
      return _result
    except _core._FallbackException:
      return io_write_kafka_eager_fallback(
          message, topic, servers, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + ' name: ' + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def io_write_kafka_eager_fallback(message, topic, servers, name=None, ctx=None):
  """This is the slowpath function for Eager mode.

  This is for function io_write_kafka
  """
  _ctx = ctx if ctx else _context.context()
  message = _ops.convert_to_tensor(message, _dtypes.string)
  topic = _ops.convert_to_tensor(topic, _dtypes.string)
  servers = _ops.convert_to_tensor(servers, _dtypes.string)
  _inputs_flat = [message, topic, servers]
  _attrs = None
  _result = _execute.execute(
      b'IOWriteKafka',
      1,
      inputs=_inputs_flat,
      attrs=_attrs,
      ctx=_ctx,
      name=name)
  _execute.record_gradient('IOWriteKafka', _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result
