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
# """Evaluation of Top k hitrate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os

import graphlearn as gl
import tensorflow as tf

from easy_rec.python.protos.dataset_pb2 import DatasetConfig
from easy_rec.python.utils import config_util
from easy_rec.python.utils.config_util import process_multi_file_input_path
from easy_rec.python.utils.hit_rate_utils import compute_hitrate_batch
from easy_rec.python.utils.hit_rate_utils import load_graph
from easy_rec.python.utils.hit_rate_utils import reduce_hitrate
from easy_rec.python.utils.hive_utils import HiveUtils

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

from easy_rec.python.utils.distribution_utils import set_tf_config_and_get_train_worker_num_on_ds  # NOQA

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

tf.app.flags.DEFINE_string('item_emb_table', '', 'item embedding table name')
tf.app.flags.DEFINE_string('gt_table', '', 'ground truth table name')
tf.app.flags.DEFINE_string('hitrate_details_result', '',
                           'hitrate detail file path')
tf.app.flags.DEFINE_string('total_hitrate_result', '',
                           'total hitrate result file path')

tf.app.flags.DEFINE_string('pipeline_config_path', '', 'pipeline config path')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'embedding dimension')
tf.app.flags.DEFINE_string('recall_type', 'i2i', 'i2i or u2i')
tf.app.flags.DEFINE_integer('top_k', '5', 'top_k hitrate.')
tf.app.flags.DEFINE_integer('knn_metric', '0', '0(l2) or 1(ip).')
tf.app.flags.DEFINE_bool('knn_strict', False, 'use exact search.')
tf.app.flags.DEFINE_integer('timeout', '60', 'timeout')
tf.app.flags.DEFINE_integer('num_interests', 1, 'max number of interests')
tf.app.flags.DEFINE_string('gt_table_field_sep', '\t', 'gt_table_field_sep')
tf.app.flags.DEFINE_string('item_emb_table_field_sep', '\t',
                           'item_emb_table_field_sep')
tf.app.flags.DEFINE_bool('is_on_ds', False, help='is on ds')

FLAGS = tf.app.flags.FLAGS


def compute_hitrate(g, gt_all, hitrate_writer, gt_table=None):
  """Compute hitrate of each worker.

  Args:
    g: a GL Graph instance.
    gt_reader: reader of input trigger_items_table.
    hitrate_writer: writer of hitrate table.
    gt_table: ground truth table.

  Returns:
    total_hits: total hits of this worker.
    total_gt_count: total count of ground truth items of this worker.
  """
  total_hits = 0.0
  total_gt_count = 0.0

  for gt_record in gt_all:
    gt_record = list(gt_record)
    hits, gt_count, src_ids, recall_ids, recall_distances, hitrates, bad_cases, bad_dists = \
        compute_hitrate_batch(g, gt_record, FLAGS.emb_dim, FLAGS.num_interests, FLAGS.top_k)
    total_hits += hits
    total_gt_count += gt_count

    src_ids = [str(ids) for ids in src_ids]
    hitrates = [str(hitrate) for hitrate in hitrates]
    topk_recalls = [','.join(str(x) for x in ids) for ids in recall_ids]
    topk_dists = [
        ','.join('|'.join(str(x)
                          for x in dist)
                 for dist in dists)
        for dists in recall_distances
    ]
    bad_cases = [','.join(str(x) for x in bad_case) for bad_case in bad_cases]
    bad_dists = [','.join(str(x) for x in dist) for dist in bad_dists]

    hitrate_writer.write('\n'.join([
        '\t'.join(line) for line in zip(src_ids, topk_recalls, topk_dists,
                                        hitrates, bad_cases, bad_dists)
    ]))
  print('total_hits: ', total_hits)
  print('total_gt_count: ', total_gt_count)
  return total_hits, total_gt_count


def gt_hdfs(gt_table, batch_size, gt_file_sep):

  if '*' in gt_table or ',' in gt_table:
    file_paths = tf.gfile.Glob(gt_table.split(','))
  elif tf.gfile.IsDirectory(gt_table):
    file_paths = tf.gfile.Glob(os.path.join(gt_table, '*'))
  else:
    file_paths = tf.gfile.Glob(gt_table)

  batch_list, i = [], 0
  for file_path in file_paths:
    with tf.gfile.GFile(file_path, 'r') as fin:
      for gt in fin:
        i += 1
        gt_list = gt.strip().split(gt_file_sep)
        # make id , emb_num to int
        gt_list[0], gt_list[3] = int(gt_list[0]), int(gt_list[3])
        batch_list.append(tuple(i for i in gt_list))
        if i >= batch_size:
          yield batch_list
          batch_list, i = [], 0
  if i != 0:
    yield batch_list


def main():
  tf_config = json.loads(os.environ['TF_CONFIG'])
  worker_count = len(tf_config['cluster']['worker'])
  task_index = tf_config['task']['index']
  job_name = tf_config['task']['type']

  hitrate_details_result = FLAGS.hitrate_details_result
  total_hitrate_result = FLAGS.total_hitrate_result
  i_emb_table = FLAGS.item_emb_table
  gt_table = FLAGS.gt_table

  pipeline_config = config_util.get_configs_from_pipeline_file(
      FLAGS.pipeline_config_path)
  logging.info('i_emb_table %s', i_emb_table)

  input_type = pipeline_config.data_config.input_type
  input_type_name = DatasetConfig.InputType.Name(input_type)
  if input_type_name == 'CSVInput':
    i_emb_table = process_multi_file_input_path(i_emb_table)
  else:
    hive_utils = HiveUtils(
        data_config=pipeline_config.data_config,
        hive_config=pipeline_config.hive_train_input)
    i_emb_table = hive_utils.get_table_location(i_emb_table)

  g = load_graph(i_emb_table, FLAGS.emb_dim, FLAGS.knn_metric, FLAGS.timeout,
                 FLAGS.knn_strict)
  gl.set_tracker_mode(0)
  gl.set_field_delimiter(FLAGS.item_emb_table_field_sep)

  cluster = tf.train.ClusterSpec({
      'ps': tf_config['cluster']['ps'],
      'worker': tf_config['cluster']['worker']
  })
  server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

  if job_name == 'ps':
    server.join()
  else:
    worker_hosts = [
        str(host.split(':')[0]) + ':888' + str(i)
        for i, host in enumerate(tf_config['cluster']['worker'])
    ]
    worker_hosts = ','.join(worker_hosts)
    g.init(task_index=task_index, task_count=worker_count, hosts=worker_hosts)
    # Your model, use g to do some operation, such as sampling

    if input_type_name == 'CSVInput':
      gt_all = gt_hdfs(gt_table, FLAGS.batch_size, FLAGS.gt_table_field_sep)
    else:
      gt_reader = HiveUtils(
          data_config=pipeline_config.data_config,
          hive_config=pipeline_config.hive_train_input,
          selected_cols='*')
      gt_all = gt_reader.hive_read_lines(gt_table, FLAGS.batch_size)
    if not tf.gfile.IsDirectory(hitrate_details_result):
      tf.gfile.MakeDirs(hitrate_details_result)
    hitrate_details_result = os.path.join(hitrate_details_result,
                                          'part-%s' % task_index)
    details_writer = tf.gfile.GFile(hitrate_details_result, 'w')
    print('Start compute hitrate...')
    total_hits, total_gt_count = compute_hitrate(g, gt_all, details_writer,
                                                 gt_table)
    var_total_hitrate, var_worker_count = reduce_hitrate(
        cluster, total_hits, total_gt_count, task_index)

    with tf.train.MonitoredTrainingSession(
        master=server.target, is_chief=(task_index == 0)) as sess:
      outs = sess.run([var_total_hitrate, var_worker_count])

    # write after all workers have completed the calculation of hitrate.
    print('outs: ', outs)
    if outs[1] == worker_count:
      logging.info(outs)
      with tf.gfile.GFile(total_hitrate_result, 'w') as total_writer:
        total_writer.write(str(outs[0]))

    details_writer.close()
    g.close()
    print('Compute hitrate done.')


if __name__ == '__main__':
  main()
