# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from datetime import datetime

import common_io
import numpy as np
import tensorflow as tf

try:
  import graphlearn as gl
except:  # noqa: E722
  logging.warning(
      'GraphLearn is not installed. You can install it by "pip install https://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/graphlearn-0.7-cp27-cp27mu-linux_x86_64.whl.'  # noqa: E501
  )

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class VectorRetrieve(object):

  def __init__(self,
               query_table,
               doc_table,
               out_table,
               ndim,
               delimiter=',',
               batch_size=4,
               index_type='ivfflat',
               nlist=10,
               nprobe=2,
               distance=1,
               m=8):
    """Retrieve top n neighbours by query vector.

    Args:
      query_table: query vector table
      doc_table: document vector table
      out_table: output table
      ndim: int, number of feature dimensions
      delimiter: delimiter for feature vectors
      batch_size: query batch size
      index_type: search model `flat`, `ivfflat`, `ivfpq`, `gpu_ivfflat`
      nlist: number of split part on each worker
      nprobe: probe part on each worker
      distance: type of distance,  0 is l2 distance(default), 1 is inner product.
      m: number of dimensions for each node after compress
    """
    self.query_table = query_table
    self.doc_table = doc_table
    self.out_table = out_table
    self.ndim = ndim
    self.delimiter = delimiter
    self.batch_size = batch_size

    gl.set_inter_threadnum(8)
    gl.set_knn_metric(distance)
    knn_option = gl.IndexOption()
    knn_option.name = 'knn'
    knn_option.index_type = index_type
    knn_option.nlist = nlist
    knn_option.nprobe = nprobe
    knn_option.m = m
    self.knn_option = knn_option

  def __call__(self, top_n, task_index, task_count, *args, **kwargs):
    g = gl.Graph()
    g.node(
        self.doc_table,
        'doc',
        decoder=gl.Decoder(
            attr_types=['float'] * self.ndim, attr_delimiter=self.delimiter),
        option=self.knn_option)
    g.init(task_index=task_index, task_count=task_count)

    query_reader = common_io.table.TableReader(
        self.query_table, slice_id=task_index, slice_count=task_count)
    num_records = query_reader.get_row_count()
    total_batch_num = num_records // self.batch_size + 1.0
    batch_num = 0
    print('total input records: {}'.format(query_reader.get_row_count()))
    print('total_batch_num: {}'.format(total_batch_num))
    print('output_table: {}'.format(self.out_table))

    output_table_writer = common_io.table.TableWriter(self.out_table,
                                                      task_index)
    count = 0
    while True:
      try:
        batch_query_nodes, batch_query_feats = zip(
            *query_reader.read(self.batch_size, allow_smaller_final_batch=True))
        batch_num += 1.0
        print('{} process: {:.2f}'.format(datetime.now().time(),
                                          batch_num / total_batch_num))
        feats = to_np_array(batch_query_feats, self.delimiter)
        rt_ids, rt_dists = g.search('doc', feats, gl.KnnOption(k=top_n))

        for query_node, nodes, dists in zip(batch_query_nodes, rt_ids,
                                            rt_dists):
          query = np.array([query_node] * len(nodes), dtype='int64')
          output_table_writer.write(
              zip(query, nodes, dists), (0, 1, 2), allow_type_cast=False)
          count += 1
          if np.mod(count, 100) == 0:
            print('write ', count, ' query nodes totally')
      except Exception as e:
        print(e)
        break

    print('==finished==')
    output_table_writer.close()
    query_reader.close()
    g.close()


def to_np_array(batch_query_feats, attr_delimiter):
  return np.array(
      [map(float, feat.split(attr_delimiter)) for feat in batch_query_feats],
      dtype='float32')
