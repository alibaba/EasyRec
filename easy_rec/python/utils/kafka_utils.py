# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.


def kafka_maker():
  """Return a kafka dataset maker, a kafka dataset adaptation implement."""
  try:
    from easy_rec.python.input.kafka_dataset import KafkaDatasetV2
    return KafkaDatasetV2
  except Exception:
    try:
      from tensorflow_io.kafka import KafkaDataset
      return KafkaDataset
    except Exception:
      from tensorflow.python.ops import kafka_dataset_ops
      return kafka_dataset_ops.KafkaDataset
  return None


def kafka_input_params_json_parse(params, task_num, task_id, parse_type='train'):
  """Return a json object.

  Form params str fetch and build kafka offset config for current worker

  Args:
    params: a `str`, kakfa dataset offsets, i.e: '0:1,2:3'
    task_num: total task number in cluster
    task_id: current task id
    parse_type: 'train' or 'eval', default train

  Returns:
    a json object

  Raises:
    ValueError, if:
      * paratitions if offsets not equal task num
      * partiition or offset can not convert to int
    AssertionError, if:
      * parse type not in 'train' or 'eval'
  """
  assert parse_type in ('train', 'eval')
  offsets = params.split(",")
  offsets = [o.split(":") for o in offsets if len(o.split(":")) == 2]
  offsets = {int(o[0]): int(o[1]) for o in offsets}
  if len(offsets) < task_num:
    raise ValueError('kakfa offsets partitions should be bigger than task number')
  try:
    offset = offsets[task_id]
    if parse_type == 'train':
      return {'kafka_train_input.offset': offset}
    elif parse_type == 'eval':
      return {'kafka_eval_input.offset': [offset], 'kafka_eval_input.partitions': 1}
  except KeyError as e:
    raise ValueError('kafka offsets partition must be equals task id')

