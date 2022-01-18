# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================
"""Evaluation of Top k hitrate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graphlearn as gl
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('task_index', None, 'Task index')
flags.DEFINE_integer('task_count', None, 'Task count')
flags.DEFINE_string('job_name', None, 'worker or ps or aligraph')
flags.DEFINE_string('ps_hosts', '', 'ps hosts')
flags.DEFINE_string('worker_hosts', '', 'worker hosts')
flags.DEFINE_string('tables', '', 'input odps tables name')
flags.DEFINE_string('outputs', '', 'ouput odps tables name')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('emb_dim', 128, 'embedding dimension')
flags.DEFINE_string('recall_type', 'i2i', 'i2i or u2i')
flags.DEFINE_integer('top_k', '5', 'top_k hitrate.')
flags.DEFINE_integer('knn_metric', '0', '0(l2) or 1(ip).')
flags.DEFINE_bool('knn_strict', False, 'use exact search.')
flags.DEFINE_integer('timeout', '60', 'timeout')
flags.DEFINE_integer('num_interests', 1, 'max number of interests')


def load_graph(i_emb_table):
  """Load embedding tables in GL that used to lookup embedding and do knn search."""
  gl.set_knn_metric(FLAGS.knn_metric)
  gl.set_timeout(FLAGS.timeout)
  option = gl.IndexOption()
  option.name = 'knn'
  if FLAGS.knn_strict:
    # option.index_type = "flat"
    option.index_type = 'ivfflat'
    option.nlist = 5
    option.nprobe = 5
  else:
    option.index_type = 'ivfflat'
    option.nlist = 5
    option.nprobe = 2
  g = gl.Graph().node(
      i_emb_table,
      node_type='i',
      decoder=gl.Decoder(
          attr_types=['float'] * FLAGS.emb_dim, attr_delimiter=','),
      option=option)
  return g


def batch_hitrate(src_ids, recall_ids, recall_distances, gt_items, mask=None):
  """Compute hitrate of a batch of src ids.

  Args:
    src_ids: trigger id, a numpy array.
    recall_ids: recalled ids by src_ids, a numpy array.
    recall_distances: corresponding distances of recalled ids, a numpy array.
    gt_items: batch of ground truth item ids list, a list of list.
    mask: some models have different number of interests.

  Returns:
    hitrates: hitrate of src_ids, a list.
    bad_cases: bad cases, a list of list.
    bad_dsts: distances of bad cases, a list of list.
    hits: total hit counts of a batch of src ids, a scalar.
    gt_count: total ground truth items num of a batch of src ids, a scalar.
  """
  hitrates = []
  bad_cases = []
  bad_dists = []
  hits = 0.0
  gt_count = 0.0
  for idx, src_id in enumerate(src_ids):
    recall_id = recall_ids[idx]
    recall_distance = recall_distances[idx]

    hit_count = 0.0
    bad_case = []
    bad_dist = []
    gt_items_size = len(gt_items[idx])
    if gt_items_size == 0:  # just skip invalid record.
      print('Id {:d} has no related items sequence, just skip.'.format(src_id))
      continue
    for interest_id in range(FLAGS.num_interests):
      if not mask[idx, interest_id]:
        continue
      for k, id in enumerate(recall_id[interest_id]):
        if id in gt_items[idx]:
          hit_count += 1
        else:
          bad_case.append(id)
          bad_dist.append(recall_distance[interest_id][k])
    hitrates.append(hit_count / gt_items_size)
    hits += hit_count
    gt_count += gt_items_size
    bad_cases.append(bad_case)
    bad_dists.append(bad_dist)
  return hitrates, bad_cases, bad_dists, hits, gt_count


def compute_hitrate(g, gt_reader, hitrate_writer):
  """Compute hitrate of each worker.

  Args:
    g: a GL Graph instance.
    gt_reader: odps reader of input trigger_items_table.
    hitrate_writer: odps writer of hitrate table.

  Returns:
    total_hits: total hits of this worker.
    total_gt_count: total count of ground truth items of this worker.
  """
  total_hits = 0.0
  total_gt_count = 0.0
  while True:
    try:
      gt_record = gt_reader.read(FLAGS.batch_size)
      src_ids = np.array([src_items[0] for src_items in gt_record])

      def _to_float_attrs(x):
        # incase user embedding is not present
        if x == '':
          return np.zeros([FLAGS.emb_dim], dtype=np.float32)
        embed = np.array(x.split(','), dtype=np.float32)
        assert len(embed) == FLAGS.emb_dim, 'invalid embed len=%d, x=%s' % (
            len(embed), x)
        return embed

      def _to_multi_float_attrs(x, userid):
        if x == '':
          arr = [_to_float_attrs(x) for i in range(FLAGS.num_interests)]
        else:
          arr = [_to_float_attrs(sub_x) for sub_x in x.split('|')]
        assert len(
            arr
        ) == FLAGS.num_interests, 'invalid arr len=%d, x=%s, userid=%s' % (
            len(arr), x, userid)
        return arr

      user_embedding = np.array([
          _to_multi_float_attrs(src_items[2], src_items[0])
          for src_items in gt_record
      ])
      user_emb_num = [src_items[3] for src_items in gt_record]

      print('max(user_emb_num) = %d len(src_ids) = %d' %
            (np.max(user_emb_num), len(src_ids)))

      # a list of list.
      gt_items = [map(int, src_items[1].split(',')) for src_items in gt_record]

      print('src_nodes.float_attrs.shape=%s' % str(user_embedding.shape))
      user_embedding = user_embedding.reshape([-1, user_embedding.shape[-1]])
      # numpy array
      recall_ids, recall_distances = g.search('i', user_embedding,
                                              gl.KnnOption(k=FLAGS.top_k))
      print('recall_ids.shape=%s' % str(recall_ids.shape))

      def _make_mask(lens):
        mask = np.ones([len(lens), FLAGS.num_interests], dtype=np.float32)
        for tmp_id, tmp_len in enumerate(lens):
          mask[tmp_id, int(tmp_len):] = 0
        return mask

      mask = _make_mask(user_emb_num)
      recall_ids = recall_ids.reshape(
          [-1, FLAGS.num_interests, recall_ids.shape[-1]])
      recall_distances = recall_distances.reshape(
          [-1, FLAGS.num_interests, recall_distances.shape[-1]])
      hitrates, bad_cases, bad_dists, hits, gt_count = batch_hitrate(
          src_ids, recall_ids, recall_distances, gt_items, mask)

      topk_recalls = [','.join(str(x) for x in ids) for ids in recall_ids]
      topk_dists = [
          ','.join(str(x) for x in dists) for dists in recall_distances
      ]
      bad_cases = [','.join(str(x) for x in case) for case in bad_cases]
      bad_dists = [','.join(str(x) for x in dist) for dist in bad_dists]
      total_hits += hits
      total_gt_count += gt_count
      hitrate_writer.write(
          list(
              zip(src_ids, topk_recalls, topk_dists, hitrates, bad_cases,
                  bad_dists)),
          indices=[0, 1, 2, 3, 4, 5])
    except tf.python_io.OutOfRangeException:
      break
  return total_hits, total_gt_count


def reduce_hitrate(cluster, hits, count):
  """Reduce hitrate of all workers.

  Args:
    cluster: tf cluster.
    hits: total_hits of each worker.
    count: total count of ground truth items of each worker.

  Returns:
    var_total_hitrate: variable of total hitrate.
    var_worker_count: variable used to mark the number of worker that
    have completed the calculation of hitrate.
  """
  with tf.device(
      tf.train.replica_device_setter(
          worker_device='/job:worker/task:%d' % FLAGS.task_index,
          cluster=cluster)):
    with tf.variable_scope('hitrate_var', reuse=tf.AUTO_REUSE):
      var_worker_count = tf.get_variable(
          'worker_count',
          shape=(),
          dtype=tf.int32,
          initializer=tf.zeros_initializer())
      var_hits = tf.get_variable(
          'hits',
          shape=(),
          dtype=tf.float32,
          initializer=tf.zeros_initializer())
      var_gt_count = tf.get_variable(
          'gt_count',
          shape=(),
          dtype=tf.float32,
          initializer=tf.zeros_initializer())
      var_total_hitrate = tf.get_variable(
          'total_hitate',
          shape=(),
          dtype=tf.float32,
          initializer=tf.zeros_initializer())

      var_hits = tf.assign_add(var_hits, hits, use_locking=True)
      var_gt_count = tf.assign_add(var_gt_count, count, use_locking=True)
      var_gt_count = tf.Print(
          var_gt_count, [var_gt_count, var_hits],
          message='var_gt_count/var_hits')
      var_total_hitrate = tf.assign(
          var_total_hitrate, var_hits / var_gt_count, use_locking=True)
      with tf.control_dependencies([var_total_hitrate]):
        var_worker_count = tf.assign_add(var_worker_count, 1, use_locking=True)
  return var_total_hitrate, var_worker_count


def main():
  worker_count = len(FLAGS.worker_hosts.split(','))
  input_tables = FLAGS.tables.split(',')
  if FLAGS.recall_type == 'u2i':
    i_emb_table, gt_table = input_tables
    g = load_graph(i_emb_table)
  else:
    i_emb_table, gt_table = input_tables[-2], input_tables[-1]
    g = load_graph(i_emb_table)
  hitrate_details_table, total_hitrate_table = FLAGS.outputs.split(',')

  cluster = tf.train.ClusterSpec({
      'ps': FLAGS.ps_hosts.split(','),
      'worker': FLAGS.worker_hosts.split(',')
  })
  server = tf.train.Server(
      cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == 'ps':
    server.join()
  else:
    g.init(task_index=FLAGS.task_index, task_count=worker_count)
    gt_reader = tf.python_io.TableReader(
        gt_table,
        slice_id=FLAGS.task_index,
        slice_count=worker_count,
        capacity=2048)
    details_writer = tf.python_io.TableWriter(
        hitrate_details_table, slice_id=FLAGS.task_index)
    print('Start compute hitrate...')
    total_hits, total_gt_count = compute_hitrate(g, gt_reader, details_writer)
    var_total_hitrate, var_worker_count = reduce_hitrate(
        cluster, total_hits, total_gt_count)

    with tf.train.MonitoredTrainingSession(
        master=server.target, is_chief=(FLAGS.task_index == 0)) as sess:
      outs = sess.run([var_total_hitrate, var_worker_count])

    # write after all workers have completed the calculation of hitrate.
    if outs[1] == worker_count:
      with tf.python_io.TableWriter(total_hitrate_table) as total_writer:
        total_writer.write([outs[0]], indices=[0])

    gt_reader.close()
    details_writer.close()
    g.close()
    print('Compute hitrate done.')


if __name__ == '__main__':
  main()
