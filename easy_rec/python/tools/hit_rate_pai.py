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

import tensorflow as tf

from easy_rec.python.utils.hit_rate_utils import compute_hitrate_batch
from easy_rec.python.utils.hit_rate_utils import load_graph
from easy_rec.python.utils.hit_rate_utils import reduce_hitrate

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
      hits, gt_count, src_ids, recall_ids, recall_distances, hitrates, bad_cases, bad_dists = \
          compute_hitrate_batch(g, gt_record, FLAGS.emb_dim, FLAGS.num_interests, FLAGS.top_k)
      total_hits += hits
      total_gt_count += gt_count
      topk_recalls = [','.join(str(x) for x in ids) for ids in recall_ids]
      topk_dists = [
          ','.join(str(x) for x in dists) for dists in recall_distances
      ]
      bad_cases = [','.join(str(x) for x in case) for case in bad_cases]
      bad_dists = [','.join(str(x) for x in dist) for dist in bad_dists]

      hitrate_writer.write(
          list(
              zip(src_ids, topk_recalls, topk_dists, hitrates, bad_cases,
                  bad_dists)),
          indices=[0, 1, 2, 3, 4, 5])
    except tf.python_io.OutOfRangeException:
      break
  return total_hits, total_gt_count


def main():
  worker_count = len(FLAGS.worker_hosts.split(','))
  input_tables = FLAGS.tables.split(',')
  if FLAGS.recall_type == 'u2i':
    i_emb_table, gt_table = input_tables
    g = load_graph(i_emb_table, FLAGS.emb_dim, FLAGS.knn_metric, FLAGS.timeout,
                   FLAGS.knn_strict)
  else:
    i_emb_table, gt_table = input_tables[-2], input_tables[-1]
    g = load_graph(i_emb_table, FLAGS.emb_dim, FLAGS.knn_metric, FLAGS.timeout,
                   FLAGS.knn_strict)
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
        cluster, total_hits, total_gt_count, FLAGS.task_index)

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
