# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import print_function

import logging
import os

import faiss
import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

tf.app.flags.DEFINE_string('tables', '', 'tables passed by pai command')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'batch size')
tf.app.flags.DEFINE_integer('embedding_dim', 32, 'embedding dimension')
tf.app.flags.DEFINE_string('index_output_dir', '', 'index output directory')
tf.app.flags.DEFINE_string('index_type', 'IVFFlat', 'index type')
tf.app.flags.DEFINE_integer('ivf_nlist', 1000, 'nlist')
tf.app.flags.DEFINE_integer('hnsw_M', 32, 'hnsw M')
tf.app.flags.DEFINE_integer('hnsw_efConstruction', 200, 'hnsw efConstruction')
tf.app.flags.DEFINE_integer('debug', 0, 'debug index')

FLAGS = tf.app.flags.FLAGS


def main(argv):
  reader = tf.python_io.TableReader(
      FLAGS.tables, slice_id=0, slice_count=1, capacity=FLAGS.batch_size * 2)
  i = 0
  id_map_f = tf.gfile.GFile(
      os.path.join(FLAGS.index_output_dir, 'id_mapping'), 'w')
  embeddings = []
  while True:
    try:
      records = reader.read(FLAGS.batch_size)
      for j, record in enumerate(records):
        if isinstance(record[0], bytes):
          eid = record[0].decode('utf-8')
        id_map_f.write('%s\n' % eid)

      embeddings.extend(
          [list(map(float, record[1].split(b','))) for record in records])
      i += 1
      if i % 100 == 0:
        logging.info('read %d embeddings.' % (i * FLAGS.batch_size))
    except tf.python_io.OutOfRangeException:
      break
  reader.close()
  id_map_f.close()

  logging.info('Building faiss index..')
  if FLAGS.index_type == 'IVFFlat':
    quantizer = faiss.IndexFlatIP(FLAGS.embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, FLAGS.embedding_dim, FLAGS.ivf_nlist,
                               faiss.METRIC_INNER_PRODUCT)
  elif FLAGS.index_type == 'HNSWFlat':
    index = faiss.IndexHNSWFlat(FLAGS.embedding_dim, FLAGS.hnsw_M,
                                faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = FLAGS.hnsw_efConstruction
  else:
    raise NotImplementedError

  embeddings = np.array(embeddings)
  if FLAGS.index_type == 'IVFFlat':
    logging.info('train embeddings...')
    index.train(embeddings)

  logging.info('build embeddings...')
  index.add(embeddings)
  faiss.write_index(index, 'faiss_index')

  with tf.gfile.GFile(
      os.path.join(FLAGS.index_output_dir, 'faiss_index'), 'wb') as f_out:
    with open('faiss_index', 'rb') as f_in:
      f_out.write(f_in.read())

  if FLAGS.debug != 0:
    # IVFFlat
    for ivf_nlist in [100, 500, 1000, 2000]:
      quantizer = faiss.IndexFlatIP(FLAGS.embedding_dim)
      index = faiss.IndexIVFFlat(quantizer, FLAGS.embedding_dim, ivf_nlist,
                                 faiss.METRIC_INNER_PRODUCT)
      index.train(embeddings)
      index.add(embeddings)
      index_name = 'faiss_index_ivfflat_nlist%d' % ivf_nlist
      faiss.write_index(index, index_name)
      with tf.gfile.GFile(
          os.path.join(FLAGS.index_output_dir, index_name), 'wb') as f_out:
        with open(index_name, 'rb') as f_in:
          f_out.write(f_in.read())

    # HNSWFlat
    for hnsw_M in [16, 32, 64, 128]:
      for hnsw_efConstruction in [64, 128, 256, 512, 1024, 2048, 4096, 8196]:
        if hnsw_efConstruction < hnsw_M * 2:
          continue
        index = faiss.IndexHNSWFlat(FLAGS.embedding_dim, hnsw_M,
                                    faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_efConstruction
        index.add(embeddings)
        index_name = 'faiss_index_hnsw_M%d_ef%d' % (hnsw_M, hnsw_efConstruction)
        faiss.write_index(index, index_name)
        with tf.gfile.GFile(
            os.path.join(FLAGS.index_output_dir, index_name), 'wb') as f_out:
          with open(index_name, 'rb') as f_in:
            f_out.write(f_in.read())


if __name__ == '__main__':
  tf.app.run()
