# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import tensorflow as tf

from easy_rec.python.input.csv_input import CSVInput
from easy_rec.python.ops.gen_str_avx_op import str_split_by_chr

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CSVInputEx(CSVInput):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               pipeline_config=None):
    super(CSVInputEx,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)

  def _parse_csv(self, line):
    record_defaults = [
        self.get_type_defaults(t, v)
        for t, v in zip(self._input_field_types, self._input_field_defaults)
    ]

    def _check_data(line):
      sep = self._data_config.separator
      if type(sep) != type(str):
        sep = sep.encode('utf-8')
      field_num = len(line[0].split(sep))
      assert field_num == len(record_defaults), \
          'sep[%s] maybe invalid: field_num=%d, required_num=%d' % \
          (sep, field_num, len(record_defaults))
      return True

    fields = str_split_by_chr(
        line, self._data_config.separator, skip_empty=False)
    tmp_fields = tf.reshape(fields.values, [-1, len(record_defaults)])
    fields = []
    for i in range(len(record_defaults)):
      if type(record_defaults[i]) == int:
        fields.append(
            tf.string_to_number(
                tmp_fields[:, i], tf.int64, name='field_as_int_%d' % i))
      elif type(record_defaults[i]) in [float, np.float32, np.float64]:
        fields.append(
            tf.string_to_number(
                tmp_fields[:, i], tf.float32, name='field_as_flt_%d' % i))
      elif type(record_defaults[i]) in [str, type(u''), bytes]:
        fields.append(tmp_fields[:, i])
      elif type(record_defaults[i]) == bool:
        fields.append(
            tf.logical_or(
                tf.equal(tmp_fields[:, i], 'True'),
                tf.equal(tmp_fields[:, i], 'true')))
      else:
        assert 'invalid types: %s' % str(type(record_defaults[i]))

    keep_ids = [
        self._input_fields.index(x)
        for x in self._label_fields + self._effective_fields
    ]
    inputs = {self._input_fields[x]: fields[x] for x in keep_ids}

    return inputs
