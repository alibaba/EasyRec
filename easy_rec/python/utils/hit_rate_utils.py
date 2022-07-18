import logging

import graphlearn as gl
import numpy as np
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def load_graph(i_emb_table, emb_dim, knn_metric, timeout, knn_strict):
  """Load embedding tables in GL.

  that used to lookup embedding and do knn search.
  """
  gl.set_knn_metric(knn_metric)
  gl.set_timeout(timeout)
  option = gl.IndexOption()
  option.name = 'knn'
  if knn_strict:
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
      decoder=gl.Decoder(attr_types=['float'] * emb_dim, attr_delimiter=','),
      option=option)
  return g


def batch_hitrate(src_ids,
                  recall_ids,
                  recall_distances,
                  gt_items,
                  num_interests,
                  mask=None):
  """Compute hitrate of a batch of src ids.

  Args:
    src_ids: trigger id, a numpy array.
    recall_ids: recalled ids by src_ids, a numpy array.
    recall_distances: corresponding distances of recalled ids, a numpy array.
    gt_items: batch of ground truth item ids list, a list of list.
    num_interests: max number of interests.
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

    bad_case = {}
    gt_items_size = len(gt_items[idx])
    hit_ids = []
    if gt_items_size == 0:  # just skip invalid record.
      print('Id {:d} has no related items sequence, just skip.'.format(src_id))
      continue
    for interest_id in range(num_interests):
      if not mask[idx, interest_id]:
        continue
      for k, id in enumerate(recall_id[interest_id]):
        if id in gt_items[idx]:
          if id not in hit_ids:
            hit_ids.append(id)
        else:
          dis = recall_distance[interest_id][k]
          if id not in bad_case:
            bad_case[id] = dis
          elif dis < bad_case[id]:
            bad_case[id] = dis
    hit_count = float(len(hit_ids))
    hitrates.append(hit_count / gt_items_size)
    hits += hit_count
    gt_count += gt_items_size
    bad_cases.append([x for x in bad_case])
    bad_dists.append([bad_case[x] for x in bad_case])
  return hitrates, bad_cases, bad_dists, hits, gt_count


def reduce_hitrate(cluster, hits, count, task_index):
  """Reduce hitrate of all workers.

  Args:
    cluster: tf cluster.
    hits: total_hits of each worker.
    count: total count of ground truth items of each worker.
    task_index: worker index.

  Returns:
    var_total_hitrate: variable of total hitrate.
    var_worker_count: variable used to mark the number of worker that
    have completed the calculation of hitrate.
  """
  with tf.device(
      tf.train.replica_device_setter(
          worker_device='/job:worker/task:%d' % task_index, cluster=cluster)):
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


def compute_hitrate_batch(g, gt_record, emb_dim, num_interests, top_k):
  """Reduce hitrate of one batch.

  Args:
    g: a GL Graph instance.
    gt_record: record list of groung truth.
    emb_dim: embedding dim.
    num_interests: max number of interests.
    top_k: top_k hitrate.

  Returns:
    hits: total hit counts of a batch of src ids, a scalar.
    gt_count: total ground truth items num of a batch of src ids, a scalar.
    src_ids: src ids, a list.
    recall_ids: recall ids, a list.
    recall_distances: recall distances, a list.
    hitrates: hitrate of a batch of src_ids, a list.
    bad_cases: bad cases, a list of list.
    bad_dsts: distances of bad cases, a list of list.
  """

  def _to_float_attrs(x):
    # incase user embedding is not present
    if x == '':
      return np.zeros([emb_dim], dtype=np.float32)
    embed = np.array(x.split(','), dtype=np.float32)
    assert len(embed) == emb_dim, 'invalid embed len=%d, x=%s' % (len(embed), x)
    return embed

  def _to_multi_float_attrs(x, userid):
    if x == '':
      arr = [_to_float_attrs(x) for i in range(num_interests)]
    else:
      arr = [_to_float_attrs(sub_x) for sub_x in x.split('|')]
    assert len(arr) == num_interests, 'invalid arr len=%d, x=%s, userid=%s' % (
        len(arr), x, userid)
    return arr

  src_ids = np.array([src_items[0] for src_items in gt_record])
  user_embedding = np.array([
      _to_multi_float_attrs(src_items[2], src_items[0])
      for src_items in gt_record
  ])
  user_emb_num = [src_items[3] for src_items in gt_record]

  print('max(user_emb_num) = %d len(src_ids) = %d' %
        (np.max(user_emb_num), len(src_ids)))

  # a list of list.
  gt_items = [
      list(map(int, src_items[1].split(','))) for src_items in gt_record
  ]

  logging.info('src_nodes.float_attrs.shape=%s' % str(user_embedding.shape))
  user_embedding = user_embedding.reshape([-1, user_embedding.shape[-1]])
  # numpy array
  recall_ids, recall_distances = g.search('i', user_embedding,
                                          gl.KnnOption(k=top_k))
  logging.info('recall_ids.shape=%s' % str(recall_ids.shape))

  def _make_mask(lens):
    mask = np.ones([len(lens), num_interests], dtype=np.float32)
    for tmp_id, tmp_len in enumerate(lens):
      mask[tmp_id, int(tmp_len):] = 0
    return mask

  mask = _make_mask(user_emb_num)
  recall_ids = recall_ids.reshape([-1, num_interests, recall_ids.shape[-1]])
  recall_distances = recall_distances.reshape(
      [-1, num_interests, recall_distances.shape[-1]])
  hitrates, bad_cases, bad_dists, hits, gt_count = batch_hitrate(
      src_ids, recall_ids, recall_distances, gt_items, num_interests, mask)
  return hits, gt_count, src_ids, recall_ids, recall_distances, hitrates, bad_cases, bad_dists
