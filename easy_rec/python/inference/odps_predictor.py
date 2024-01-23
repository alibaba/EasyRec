# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from easy_rec.python.inference.predictor import Predictor


class ODPSPredictor(Predictor):

  def __init__(self,
               model_path,
               fg_json_path=None,
               profiling_file=None,
               all_cols='',
               all_col_types=''):
    super(ODPSPredictor, self).__init__(model_path, profiling_file,
                                        fg_json_path)
    self._all_cols = [x.strip() for x in all_cols.split(',') if x != '']
    self._all_col_types = [
        x.strip() for x in all_col_types.split(',') if x != ''
    ]
    self._record_defaults = [
        self._get_defaults(col_name, col_type)
        for col_name, col_type in zip(self._all_cols, self._all_col_types)
    ]

  def _get_reserved_cols(self, reserved_cols):
    reserved_cols = [x.strip() for x in reserved_cols.split(',') if x != '']
    return reserved_cols

  def _parse_line(self, *fields):
    fields = list(fields)
    field_dict = {self._all_cols[i]: fields[i] for i in range(len(fields))}
    return field_dict

  def _get_dataset(self, input_path, num_parallel_calls, batch_size, slice_num,
                   slice_id):
    input_list = input_path.split(',')
    dataset = tf.data.TableRecordDataset(
        input_list,
        record_defaults=self._record_defaults,
        slice_id=slice_id,
        slice_count=slice_num,
        selected_cols=','.join(self._all_cols))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  def _get_writer(self, output_path, slice_id):
    import common_io
    table_writer = common_io.table.TableWriter(output_path, slice_id=slice_id)
    return table_writer

  def _write_lines(self, table_writer, outputs):
    assert len(outputs) > 0
    indices = list(range(0, len(outputs[0])))
    table_writer.write(outputs, indices, allow_type_cast=False)

  @property
  def out_of_range_exception(self):
    return (tf.python_io.OutOfRangeException, tf.errors.OutOfRangeError)

  def _get_reserve_vals(self, reserved_cols, output_cols, all_vals, outputs):
    reserve_vals = [all_vals[k] for k in reserved_cols] + \
                   [outputs[x] for x in output_cols]
    return reserve_vals
