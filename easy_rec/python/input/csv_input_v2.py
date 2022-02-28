# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.input.input import Input


class CSVInputV2(Input):

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1):
    super(CSVInputV2, self).__init__(data_config, feature_config, input_path,
                                     task_index, task_num)

  def _build(self, mode, params):
    if self._input_path.startswith('hdfs://'):
      # support hdfs input
      dataset = tf.data.TextLineDataset([self._input_path])
    elif self._input_path.startswith('oss://'):
      # support oss input
      access_id="***************"
      access_key="**************"
      host = "oss-cn-beijing.aliyuncs.com"

      bucket = "/".join(self._input_path.split("/")[0:3])
      file_path = "/".join(self._input_path.split("/")[3:])
      oss_bucket_root="{}\x01id={}\x02key={}\x02host={}/".format(bucket, access_id, access_key, host)

      oss_file = oss_bucket_root + file_path
      dataset = tf.data.TextLineDataset([oss_file])
    else:
      raise ValueError("input_path must start with hdfs/oss，now it's %s".format(str(self._input_path)))

    if mode == tf.estimator.ModeKeys.TRAIN:
      if self._data_config.chief_redundant:
        dataset = dataset.shard(
            max(self._task_num - 1, 1), max(self._task_index - 1, 0))
      else:
        dataset = dataset.shard(self._task_num, self._task_index)
    else:
      dataset = dataset.repeat(1)
    _NUM_PARALLEL_CALLS = 8
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    dataset = dataset.map(self._parse_csv, num_parallel_calls=_NUM_PARALLEL_CALLS)
    dataset = dataset.map(map_func=self._preprocess, num_parallel_calls=_NUM_PARALLEL_CALLS)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
