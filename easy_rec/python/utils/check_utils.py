# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def check_split(line, sep, requried_field_num, field_name=''):
  assert sep, 'must have separator.' + (' field: %s.' %
                                        field_name) if field_name else ''

  for one_line in line:
    field_num = len(one_line.split(sep))
    if field_name:
      assert_info = 'sep[%s] maybe invalid. field_num=%d, required_num=%d, field: %s, value: %s, ' \
                    'please check separator and data.' % \
                    (sep, field_num, requried_field_num, field_name, one_line)
    else:
      assert_info = 'sep[%s] maybe invalid. field_num=%d, required_num=%d, current line is: %s, ' \
                    'please check separator and data.' % \
                    (sep, field_num, requried_field_num, one_line)
    assert field_num == requried_field_num, assert_info
  return True


def check_string_to_number(field_vals, field_name):
  for val in field_vals:
    try:
      float(val)
    except:  # noqa: E722
      assert False, 'StringToNumber ERROR: cannot convert string_to_number, field: %s, value: %s. ' \
                    'please check data.' % (field_name, val)
  return True


def check_sequence(pipeline_config_path, features):
  seq_att_groups = pipeline_config_path.model_config.seq_att_groups
  if not seq_att_groups:
    return
  for seq_att_group in seq_att_groups:
    seq_att_maps = seq_att_group.seq_att_map
    if not seq_att_maps:
      return
    for seq_att_map in seq_att_maps:
      assert len(seq_att_map.key) == len(seq_att_map.hist_seq), \
          'The size of hist_seq must equal to the size of key in one seq_att_map.'
      size_list = []
      for hist_seq in seq_att_map.hist_seq:
        cur_seq_size = len(features[hist_seq].values)
        size_list.append(cur_seq_size)
      hist_seqs = ' '.join(seq_att_map.hist_seq)
      assert len(set(size_list)) == 1, \
          'SequenceFeature Error: The size in [%s] should be consistent. Please check input: [%s].' % \
          (hist_seqs, hist_seqs)


def check_env_and_input_path(pipeline_config, input_path):
  input_type = pipeline_config.data_config.input_type
  input_type_name = DatasetConfig.InputType.Name(input_type)
  ignore_input_list = [
      DatasetConfig.InputType.TFRecordInput,
      DatasetConfig.InputType.BatchTFRecordInput,
      DatasetConfig.InputType.KafkaInput,
      DatasetConfig.InputType.DataHubInput,
      DatasetConfig.InputType.HiveInput,
      DatasetConfig.InputType.DummyInput,
  ]
  if input_type in ignore_input_list:
    return True
  assert_info = 'Current InputType is %s, InputPath is %s. Please check InputType and InputPath.' % \
                (input_type_name, input_path)
  if input_type_name.startswith('Odps'):
    # is on pai
    for path in input_path.split(','):
      if not path.startswith('odps://'):
        assert False, assert_info
    return True
  else:
    # local or ds
    for path in input_path.split(','):
      if path.startswith('odps://'):
        assert False, assert_info
  return True
