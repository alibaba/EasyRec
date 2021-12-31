import logging

import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import layer_norm
import math
if tf.__version__ >= '2.0':
  tf = tf.compat.v1

# target attention
def target_attention(dnn_config, deep_fea, name, is_training):
  cur_id, hist_id_col, seq_len = deep_fea['key'], deep_fea[
      'hist_seq_emb'], deep_fea['hist_seq_len']

  seq_max_len = tf.shape(hist_id_col)[1]
  emb_dim = hist_id_col.shape[2]

  cur_ids = tf.tile(cur_id, [1, seq_max_len])
  cur_ids = tf.reshape(cur_ids,
                        tf.shape(hist_id_col))  # (B, seq_max_len, emb_dim)

  din_net = tf.concat(
      [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
      axis=-1)  # (B, seq_max_len, emb_dim*4)

  din_layer = dnn.DNN(dnn_config, None, name, is_training)
  din_net = din_layer(din_net)
  scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

  seq_len = tf.expand_dims(seq_len, 1)
  mask = tf.sequence_mask(seq_len)
  padding = tf.ones_like(scores) * (-2**32 + 1)
  scores = tf.where(mask, scores, padding)  # [B, 1, seq_max_len]

  # Scale
  scores = tf.nn.softmax(scores)  # (B, 1, seq_max_len)
  hist_din_emb = tf.matmul(scores, hist_id_col)  # [B, 1, emb_dim]
  hist_din_emb = tf.reshape(hist_din_emb, [-1, emb_dim])  # [B, emb_dim]
  din_output = tf.concat([hist_din_emb, cur_id], axis=1)
  return din_output
  
def attention_net(net, dim, cur_seq_len, seq_size, name):
  query_net = dnn_net(net, [dim], name + '_query')  # B, seq_len，dim
  key_net = dnn_net(net, [dim], name + '_key')
  value_net = dnn_net(net, [dim], name + '_value')
  scores = tf.matmul(
      query_net, key_net, transpose_b=True)  # [B, seq_size, seq_size]

  hist_mask = tf.sequence_mask(
      cur_seq_len, maxlen=seq_size - 1)  # [B, seq_size-1]
  cur_id_mask = tf.ones([tf.shape(hist_mask)[0], 1], dtype=tf.bool)  # [B, 1]
  mask = tf.concat([hist_mask, cur_id_mask], axis=1)  # [B, seq_size]
  masks = tf.reshape(tf.tile(mask, [1, seq_size]),
                      (-1, seq_size, seq_size))  # [B, seq_size, seq_size]
  padding = tf.ones_like(scores) * (-2**32 + 1)
  scores = tf.where(masks, scores, padding)  # [B, seq_size, seq_size]

  # Scale
  scores = tf.nn.softmax(scores)  # (B, seq_size, seq_size)
  att_res_net = tf.matmul(scores, value_net)  # [B, seq_size, emb_dim]
  return att_res_net
  
def dnn_net(net, dnn_units, name):
  with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
    for idx, units in enumerate(dnn_units):
      net = tf.layers.dense(
          net, units=units, activation=tf.nn.relu, name='%s_%d' % (name, idx))
  return net

def add_and_norm(net_1, net_2, emb_dim):
  net = tf.add(net_1, net_2)
  layer = layer_norm.LayerNormalization(emb_dim)
  net = layer(net)
  return net

def multi_head_att_net(id_cols, head_count, emb_dim, seq_len, seq_size):
  multi_head_attention_res = []
  part_cols_emd_dim = int(math.ceil(emb_dim / head_count))
  for start_idx in range(0, emb_dim, part_cols_emd_dim):
    if start_idx + part_cols_emd_dim > emb_dim:
      part_cols_emd_dim = emb_dim - start_idx
    part_id_col = tf.slice(id_cols, [0, 0, start_idx],
                            [-1, -1, part_cols_emd_dim])
    part_attention_net = attention_net(
        part_id_col,
        part_cols_emd_dim,
        seq_len,
        seq_size,
        name='multi_head_%d' % start_idx)
    multi_head_attention_res.append(part_attention_net)
  multi_head_attention_res_net = tf.concat(multi_head_attention_res, axis=2)
  multi_head_attention_res_net = dnn_net(
      multi_head_attention_res_net, [emb_dim], name='multi_head_attention')
  return multi_head_attention_res_net

def self_attention(deep_fea, seq_size, head_count):
  cur_id, hist_id_col, seq_len = deep_fea['key'], deep_fea[
      'hist_seq_emb'], deep_fea['hist_seq_len']

  cur_batch_max_seq_len = tf.shape(hist_id_col)[1]

  hist_id_col = tf.cond(
      tf.constant(seq_size) > cur_batch_max_seq_len, lambda: tf.pad(
          hist_id_col, [[0, 0], [0, seq_size - cur_batch_max_seq_len - 1],
                        [0, 0]], 'CONSTANT'),
      lambda: tf.slice(hist_id_col, [0, 0, 0], [-1, seq_size - 1, -1]))
  all_ids = tf.concat([hist_id_col, tf.expand_dims(cur_id, 1)],
                      axis=1)  # b, seq_size, emb_dim

  emb_dim = int(all_ids.shape[2])
  attention_net = multi_head_att_net(all_ids, head_count, emb_dim,
                                          seq_len, seq_size)

  tmp_net = add_and_norm(
      all_ids, attention_net, emb_dim)
  feed_forward_net = dnn_net(tmp_net, [emb_dim], 'feed_forward_net')
  net = add_and_norm(
      tmp_net, feed_forward_net, emb_dim)
  atten_output = tf.reshape(net, [-1, seq_size * emb_dim])
  return atten_output


