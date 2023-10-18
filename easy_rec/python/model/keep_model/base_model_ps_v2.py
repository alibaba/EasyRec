#!/usr/bin/env python
# encoding: utf-8
"""usage dcn 两输出模型，concat none 特征."""
import json
import logging
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from base.context import Context
from base.context import conf_overwrite
from base.dataset import QDataSet
from base.qbox_wrapper import QWrapper
from layers.dnn import DNN
from layers.sequence import AttentionSequencePoolingLayer
from metric.auc import AUCUtil
from metric.bucket_auc import BucketAUCUtil
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.basic_utils import *
from utils.mysql_util import MySQLUtils
from utils.ps_deploy import deploy_ps_data


# experimental_relax_shapes: True, avoid retracing
@tf.function(experimental_relax_shapes=True)
def train_step(model, input_dic, ctx):
  """docstring."""
  with tf.GradientTape() as tape:
    output_dic = model(input_dic, training=True)

    ps_loss = output_dic['ps_loss']
    ps_grad = tf.gradients(ps_loss, [input_dic['ps_embed']])[0]

    b1_power = input_dic['b1_power']
    b2_power = input_dic['b2_power']
    #tf.print(b1_power, b2_power)
    lr_t = ctx.learning_rate * \
        tf.sqrt(1. - b2_power) / (1. - b1_power)
    tmp_m = ctx.adam_beta1 * input_dic['ps_embed_m'] + \
        (1 - ctx.adam_beta1) * ps_grad
    tmp_v = ctx.adam_beta2 * input_dic['ps_embed_v'] + \
        (1 - ctx.adam_beta2) * ps_grad * ps_grad
    tmp_w = input_dic['ps_embed'] - lr_t * tmp_m / \
        (tf.sqrt(tmp_v) + ctx.adam_epsilon)

    #tmp_w = input_dic["ps_embed"] - ctx.learning_rate * ps_grad
    result = tf.concat([tmp_w, tmp_m, tmp_v], axis=0)
    output_dic['ps_grad'] = result
    nan_flag = tf.reduce_sum(tf.cast(tf.math.is_nan(result), tf.float32))
    output_dic['nan_flag'] = nan_flag

    loss_value = output_dic['loss']
    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

  return output_dic


def make_pos(indice_list, len_list):
  """ indice_list: [0, 1, 1, 2, 2, 2]
        len_list: [1, 3, 6]
    """
  # [6]
  num = tf.shape(indice_list)[0]
  # [6, 5, 4, 3, 2, 1]
  cum_sum = tf.range(start=num, limit=0, delta=-1)

  # [5, 3, 0]
  cum_sum_of_segment_sum = num - len_list
  # [5, 3, 3, 0, 0, 0]
  cum_sum_of_segment_sum = tf.gather(cum_sum_of_segment_sum, indice_list)

  # [1, 2, 1, 3, 2, 1] - 1 = [0, 1, 0, 2, 1, 0]
  return cum_sum - cum_sum_of_segment_sum - 1


def make_sparse_indice(summed):
  """https://stackoverflow.com/questions/54790000/calculate-segment-ids-needed-in-tf-math-segment-sum-by-length-of-segments-in-ten summed is result of tf.cumsum.

  Args:
      summed (TYPE): Description

  Returns:
      TYPE: Description
  """
  return tf.searchsorted(summed, tf.range(summed[-1]), side='right')


def model_inputs_prepare(inputs, ctx):
  """Summary.

  Args:
      inputs (TYPE): Description
      ctx (TYPE): Description

  Returns:
      TYPE: Description
  """
  batch_split = ctx.batch_size // ctx.num_worker

  # slice len
  len_arr = []
  off_arr = [0]
  size_arr = []
  for idx in range(len(ctx.slot_size_vec)):
    len_arr.append(
        tf.reshape(
            tf.slice(inputs['sparse_len'], [0, ctx.slot_offset_vec[idx]],
                     [batch_split, ctx.slot_size_vec[idx]]), [-1]))
    size_arr.append(len_arr[-1][-1])
    off_arr.append(off_arr[-1] + size_arr[-1])

  # slice id
  tmp_arr = []
  for idx in range(len(ctx.slot_size_vec)):
    tmp_arr.append(
        tf.slice(inputs['sparse_id'], [off_arr[idx]], [size_arr[idx]]))

  id_arr = []
  for idx in range(len(ctx.slot_size_vec)):
    id_arr.append(
        tf.clip_by_value(tmp_arr[idx] - inputs['uniq_num_by_dim'][idx], 0,
                         2**30))

  # slice embed
  embed_arr = []
  for idx in range(len(ctx.embed_dim_vec)):
    tmp = tf.slice(
        inputs['ps_embed'], [inputs['uniq_off_by_dim'][idx]],
        [inputs['uniq_off_by_dim'][idx + 1] - inputs['uniq_off_by_dim'][idx]
         ])  # noqa
    embed_arr.append(tf.reshape(tmp, [-1, ctx.embed_dim_vec[idx]]))

  return embed_arr, id_arr, len_arr, tmp_arr


class ExportModel(Model):

  def __init__(self, model_conf, indim, name='export_model'):
    super(ExportModel, self).__init__(name=name)
    self.label_conf_list = model_conf['label']
    self.indim = indim
    self._set_inputs([
        tf.TensorSpec([None] + tmp, tf.float32, name='input' + str(i))
        for i, tmp in enumerate(indim)
    ])
    self.dnn = DNN([300, 150, 50],
                   activation='relu',
                   use_bn=True,
                   output_activation='relu')
    self.din_layer = AttentionSequencePoolingLayer()
    self.output_layers = [
        Dense(
            1, activation='sigmoid', name='output_' + label_conf['input_name'])
        for label_conf in self.label_conf_list
    ]

  def call(self, embed_input, training=False):
    target_fea = embed_input[0]

    listclick = embed_input[1]
    listclick_len = tf.reshape(listclick[..., :1], [-1, self.indim[1][0]])
    len_mask = tf.where(
        tf.equal(listclick_len, 0),
        tf.ones_like(listclick_len, dtype=tf.float32),
        tf.zeros_like(listclick_len, dtype=tf.float32))
    listclick_len = tf.cast(
        tf.reduce_sum(len_mask, axis=1, keepdims=True), tf.int32)

    listcomplete = embed_input[2]
    listcomplete_len = tf.reshape(listcomplete[..., :1], [-1, self.indim[2][0]])
    len_mask = tf.where(
        tf.equal(listcomplete_len, 0),
        tf.ones_like(listcomplete_len, dtype=tf.float32),
        tf.zeros_like(listcomplete_len, dtype=tf.float32))
    listcomplete_len = tf.cast(
        tf.reduce_sum(len_mask, axis=1, keepdims=True), tf.int32)

    click_att_data = self.din_layer([target_fea, listclick, listclick_len])
    click_att_data = tf.squeeze(click_att_data, axis=[1])

    complete_att_data = self.din_layer(
        [target_fea, listcomplete, listcomplete_len])
    complete_att_data = tf.squeeze(complete_att_data, axis=[1])

    embed = tf.concat([click_att_data, complete_att_data, embed_input[3]],
                      axis=1)
    embed = self.dnn(embed)
    outputs = [output_layer(embed) for output_layer in self.output_layers]
    return outputs


class BaseModel:

  def __init__(self, conf):
    """初始化参数.

    :param worker_list: 指定的 worker 节点列表
    :param embed_share: 0 全独立 1 field内共享 2 全共享
    """
    self.monitor_table = 'model_metrics_monitor'
    self.mysql_utils = MySQLUtils().master
    self.data_conf = conf['data']
    self.model_conf = conf['model']
    self.ps_conf = conf.get('ps', {})
    self.model_name = self.model_conf['model_name']
    self.epochs = self.data_conf.get('epochs', 1)  # 总训练轮数
    self.log_step_num = self.data_conf.get('log_step_num', 1)
    self.save_path = self.data_conf.get('save_path').rstrip(
        '/') + '/' + self.model_name + '/'
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    self.save_hdfs_path = self.data_conf.get('save_hdfs_path').rstrip(
        '/') + '/' + self.model_name + '/'
    self.save_iter = 1

    self.load_path = self.data_conf.get('load_path')
    self.model = None
    self.mode = self.model_conf['mode']
    self.profile = self.model_conf.get('profile', False)
    self.tb_callback = None
    if self.profile:
      self.tb_callback = tf.keras.callbacks.TensorBoard(
          log_dir=self.save_path + '/profile', profile_batch='10,12')

    if self.mode == 'Update':
      self.process_update_conf()
    self.feature_conf, self.group_length = self.parse_feature_conf(
        conf['feature'])
    self.custom_group_count = len(self.group_length) - 1
    self.export_model_class = self.get_export_model_classs()

    if self.data_conf.get('deploy'):
      shell(f'hadoop fs -mkdir -p {self.save_hdfs_path}')

  def parse_feature_conf(self, conf_list):
    group_features = {}
    group_len = {}
    for conf in conf_list:
      group = conf.get('group', 'default')
      if group not in group_features:
        group_features[group] = []
        group_len[group] = conf.get('length', 1)
      group_features[group].append(conf)

    if len(group_features) == 1:
      return conf_list, None
    return group_features, group_len

  def get_export_model_classs(self):
    return ExportModel

  def process_update_conf(self):
    self.mode = 'Train'
    self.data_conf['deploy'] = True
    model_info_path = self.save_hdfs_path + 'model_info.json'
    model_info_local_path = self.save_path + 'model_info.json'
    if os.path.exists(model_info_local_path):
      shell(f'rm -f {model_info_local_path}')
    shell(f'hadoop fs -get {model_info_path} {model_info_local_path}')
    recent_date = ''
    with open(model_info_local_path, 'r') as target:
      model_info = json.load(target)
      recent_date = model_info['date']

    recent_dir = self.save_hdfs_path + recent_date
    next_day = get_next_day(recent_date)
    self.data_conf['train_end_day'] = next_day
    self.data_conf['train_days'] = 1
    self.data_conf['eval_day'] = next_day
    self.data_conf['load_path'] = self.save_path + recent_date
    self.data_conf['load_embed_only'] = False
    self.load_path = self.data_conf['load_path']
    print('download model data', recent_date, self.load_path)
    if os.path.exists(self.load_path):
      shell(f'rm -rf {self.load_path}')
    rst = shell(f'hadoop fs -get {recent_dir} {self.load_path}')
    logging.info('download model file rst:' + str(rst))
    logging.info(f'update mode data conf {self.data_conf}')

  def build_input(self):
    """构建模型输入."""
    ctx = self.ctx
    in_dim = np.ma.innerproduct(ctx.slot_size_vec[self.custom_group_count:],
                                ctx.embed_dim_vec[self.custom_group_count:])
    if ctx.use_acc_ctr:
      in_dim += ctx.slot_num
    self.ctx.net_input_dim = in_dim
    input_dic = {}
    input_dic['sample_idx'] = Input(dtype=tf.int64, batch_shape=(None,))
    input_dic['label'] = Input(
        dtype=tf.float32, batch_shape=(None, len(self.model_conf['label'])))
    input_dic['sparse_len'] = Input(
        dtype=tf.int32, batch_shape=(None, None), name='sparse_len')
    input_dic['dense_fea'] = Input(
        dtype=tf.float32, batch_shape=(None, None), name='dense_fea')

    input_dic['uniq_num_by_dim'] = Input(dtype=tf.int32, batch_shape=(None,))
    input_dic['uniq_off_by_dim'] = Input(dtype=tf.int32, batch_shape=(None,))
    # single worker
    input_dic['sparse_id'] = Input(dtype=tf.int32, batch_shape=(None,))
    input_dic['ps_embed'] = Input(dtype=tf.float32, batch_shape=(None,))
    input_dic['ps_embed_m'] = Input(dtype=tf.float32, batch_shape=(None,))
    input_dic['ps_embed_v'] = Input(dtype=tf.float32, batch_shape=(None,))
    input_dic['ps_stat'] = Input(
        dtype=tf.float32, batch_shape=(None, 2), name='ps_stat')
    input_dic['b1_power'] = Input(
        dtype=tf.float32, batch_shape=(1,), name='b1_power')
    input_dic['b2_power'] = Input(
        dtype=tf.float32, batch_shape=(1,), name='b2_power')
    input_dic['test_mask'] = Input(
        dtype=tf.float32, batch_shape=(1, in_dim), name='test_mask')

    batch_split = ctx.batch_size // ctx.num_worker
    embed_arr, id_arr, len_arr, tmp_arr = model_inputs_prepare(input_dic, ctx)

    fea_dim = 11 if ctx.use_acc_ctr else 10
    gather_arr = []
    inputs_size = []

    show, click = tf.split(input_dic['ps_stat'], 2, axis=1)
    ctr = tf.reshape(click / (show + 10.), [-1, 1])

    for idx, pair in enumerate(self.group_length.items()):
      group, length = pair
      if group == 'default':
        continue
      conf_list = self.feature_conf[group]
      fea_size = len(conf_list)
      fea_dim_total = fea_size * fea_dim

      indice = make_sparse_indice(len_arr[idx])
      tmp = tf.gather(embed_arr[idx], id_arr[idx])
      if ctx.use_acc_ctr:
        acc_ctr = tf.gather(ctr, tmp_arr[idx])
        tmp = tf.concat([tmp, acc_ctr], axis=1)
      pos = make_pos(indice, len_arr[idx])
      # non_zero_indice 分别表示第i个样本 第j个pos
      non_zero_indice = tf.stack([indice, pos], axis=1)
      group_embed = tf.scatter_nd(non_zero_indice, tmp,
                                  [batch_split * fea_size, length, fea_dim])
      group_embed = tf.reshape(group_embed, [-1, fea_size, length, fea_dim])
      group_embed = tf.reshape(
          tf.transpose(group_embed, perm=[0, 2, 1, 3]),
          [-1, length, fea_dim_total])
      gather_arr.append(group_embed)
      inputs_size.append([length, fea_dim_total])

    for idx in range(self.custom_group_count, len(ctx.embed_dim_vec)):
      tmp_seg_ids = make_sparse_indice(len_arr[idx])
      tmp = tf.math.segment_sum(
          tf.gather(embed_arr[idx], id_arr[idx]), tmp_seg_ids)
      tmp = tf.reshape(
          tmp, [batch_split, ctx.slot_size_vec[idx] * ctx.embed_dim_vec[idx]])
      if ctx.use_acc_ctr:
        acc_ctr = tf.reshape(
            tf.math.segment_mean(tf.gather(ctr, tmp_arr[idx]), tmp_seg_ids),
            [batch_split, ctx.slot_size_vec[idx], 1])
        tmp = tf.reshape(
            tmp, [batch_split, ctx.slot_size_vec[idx], ctx.embed_dim_vec[idx]])
        tmp = tf.concat([tmp, acc_ctr], axis=2)
        tmp = tf.reshape(tmp, [
            batch_split, ctx.slot_size_vec[idx] * (ctx.embed_dim_vec[idx] + 1)
        ])
      gather_arr.append(tmp)

    # TODO aucdiff
    # embed_input = tf.concat(gather_arr, axis=1, name="embed_input")
    # embed_input = embed_input * input_dic["test_mask"][...,item_fea_dim_total:]
    inputs_size.append([in_dim])

    return input_dic, gather_arr, inputs_size

  def build_and_compile_model(self):
    """构建并编译模型."""
    ctx = self.ctx
    input_dic, embed_input, indim = self.build_input()
    self.export_model = self.export_model_class(self.model_conf, indim)
    # self.export_model.build((None, indim))
    # self.export_model.summary()

    y_pred = self.export_model(embed_input)
    y = input_dic['label']  # (bs, label_num)
    splited_batch_size = ctx.batch_size // ctx.num_worker

    miss_mask = tf.where(
        tf.equal(tf.reshape(input_dic['sample_idx'], [-1]), 0),
        tf.zeros([splited_batch_size], dtype=ctx.tfval),
        tf.ones([splited_batch_size], dtype=ctx.tfval))

    loss_object = tf.keras.losses.BinaryCrossentropy()
    if type(y_pred) != list:
      y_pred = [y_pred]
    total_loss = 0
    ps_loss = 0
    ttmp = tf.cast(tf.reduce_sum(miss_mask), tf.int32)
    ground_truths = []
    predictions = []
    losses = []
    for label_idx, conf in enumerate(self.model_conf['label']):
      # for ground_truth, pred, conf in zip(y, y_pred, self.model_conf['label']):
      ground_truth = y[..., label_idx]
      pred = y_pred[label_idx]
      train_space = conf.get('train_space', -1)
      if train_space >= 0:
        valid_idx = tf.where(y[..., train_space] > 0)
        ground_truth = tf.gather(ground_truth, valid_idx)
        pred = tf.gather(pred, valid_idx)
      else:
        ground_truth = ground_truth[:ttmp]
        pred = pred[:ttmp]
      ground_truth = tf.reshape(ground_truth, (-1, 1))
      pred = tf.reshape(pred, (-1, 1))
      #cur_loss = tf.nn.sigmoid_cross_entropy_with_logits(ground_truth, logit)
      #cur_loss = tf.reduce_mean(cur_loss)
      if conf.get('pos_weight'):
        pos_weight = conf.get('pos_weight', 1.)
        sample_weight = 1. + ground_truth * (pos_weight - 1.)
        cur_loss = loss_object(ground_truth, pred, sample_weight=sample_weight)
      else:
        cur_loss = loss_object(ground_truth, pred)
      losses.append(cur_loss)
      weight = conf.get('weight', 1.)
      embed_weight = conf.get('embed_weight', 1.)
      total_loss += cur_loss * weight
      ps_loss += cur_loss * embed_weight

      ground_truths.append(ground_truth)
      predictions.append(pred)

    net_dic = {
        'loss': total_loss,
        'ps_loss': ps_loss,
        'losses': losses,
        'embed_input': embed_input,
        'ground_truth': ground_truths,
        'prediction': predictions,
        'sid': input_dic['sample_idx'][:ttmp]
    }

    opt = Adam(0.001)
    model = Model(inputs=input_dic, outputs=net_dic)
    model.compile(optimizer=opt)
    self.checkpoint = tf.train.Checkpoint(
        step=tf.Variable(1), model=model, optimizer=opt)
    return model

  def load_model(self, path):
    if not self.data_conf.get('load_embed_only'):
      cpt_path = os.path.join(path, 'checkpoint')
      if not os.path.exists(cpt_path):
        raise Exception(cpt_path + ' not exists.')
      self.cpt_manager = tf.train.CheckpointManager(
          self.checkpoint, cpt_path, max_to_keep=1)
      logging.info('loading model from:{}'.format(path))
      self.checkpoint.restore(self.cpt_manager.latest_checkpoint)
    self.local_db.load()

  def save_model(self, save_date, model_info, is_deploy=None):
    path = self.save_path + save_date
    shell(f'rm -rf {path}')
    self.cpt_manager = tf.train.CheckpointManager(
        self.checkpoint, os.path.join(path, 'checkpoint'), max_to_keep=1)
    save_path = self.cpt_manager.save()
    self.local_db.save()
    logging.info('saving model to:{}'.format(save_path))
    model_info_path = self.save_path + 'model_info.json'
    with open(model_info_path, 'w') as target:
      json.dump(model_info, target)
    if is_deploy:
      #self.export_model.save(path + '/local_model', save_format="tf")
      tf.saved_model.save(self.export_model, path + '/local_model')
      hdfs_new_model_path = self.save_hdfs_path + save_date
      # deploy codis data first, make sure that when server got new version file the codis data is ready.
      deploy_ps_data(self.model_name, path, self.save_iter)
      shell(f'hadoop fs -rmr {hdfs_new_model_path}')
      logging.info('put local model to hdfs %s ' % hdfs_new_model_path)
      shell(f'hadoop fs -put {path} {hdfs_new_model_path}')
      model_info_remote_path = self.save_hdfs_path + 'model_info.json'
      shell(f'hadoop fs -put -f {model_info_path} {model_info_remote_path}')
    self.save_iter += 1

  def prepare_model(self):
    # 编译并构建模型
    self.model = self.build_and_compile_model()
    if self.load_path:
      self.load_model(self.load_path)

  def log_into_db(self, train_time, metrics):
    """log train history to db."""
    label_name = metrics['Name']
    for metric_name, metric_value in metrics.items():
      if metric_name == 'Name':
        continue
      metric_name = label_name + '_' + metric_name
      metric_time = int(time.mktime(train_time.timetuple()))
      data = {
          'model_name': self.model_name,
          'metric_time': metric_time,
          'metric_name': metric_name,
          'metric_value': float(metric_value)
      }
      where_str = f'model_name="{self.model_name}" and metric_time={metric_time} and metric_name="{metric_name}"'
      logging.info(f'log train metric into db: {data}')

      if self.mysql_utils.fetch_one(table=self.monitor_table, where=where_str):
        self.mysql_utils.update(self.monitor_table, data, where_str)
      else:
        self.mysql_utils.insert(self.monitor_table, data)

  def get_batch_input(self, batch, local_db, mask=None):
    ctx = self.ctx
    ps_stat, ps_embed, tinfo = local_db.pull(batch['task_id'])
    input_dic = {}
    # input_dic["scene_index"] = batch["scene_index"]
    input_dic['sample_idx'] = batch['sample_idx']
    input_dic['label'] = batch['label']
    input_dic['sparse_len'] = batch['sparse_len']
    input_dic['dense_fea'] = batch['dense_fea']

    input_dic['uniq_num_by_dim'] = np.cumsum(
        [0] + batch['uniq_num_by_dim'].tolist()).astype(np.int32)  # noqa
    tmp = np.array(ctx.embed_dim_vec).astype(
        np.int32) * batch['uniq_num_by_dim'].astype(np.int32)  # noqa
    input_dic['uniq_off_by_dim'] = np.cumsum([0] + tmp.tolist()).astype(
        np.int32)  # noqa
    input_dic['sparse_id'] = batch['sparse_id'][0]

    input_dic['ps_embed'] = ps_embed[0]
    input_dic['ps_embed_m'] = np.zeros_like(
        ps_embed[1]) if self.adam_restart else ps_embed[1]
    input_dic['ps_embed_v'] = np.zeros_like(
        ps_embed[2]) if self.adam_restart else ps_embed[2]
    input_dic['ps_stat'] = ps_stat
    self.b1_power *= ctx.adam_beta1
    self.b2_power *= ctx.adam_beta2
    input_dic['b1_power'] = self.b1_power
    input_dic['b2_power'] = self.b2_power
    input_dic['test_mask'] = np.ones([1, self.ctx.net_input_dim],
                                     dtype=self.ctx.npval)
    # avoid retracing of tf function.
    input_dic = {key: tf.constant(input_dic[key]) for key in input_dic}
    return input_dic

  def fit_day(self, train_itor, local_db):
    ctx = self.ctx
    batch_size = self.ctx.batch_size // self.ctx.num_worker
    metrics = [
        AUCUtil(name=conf['input_name'] + '_metric')
        for conf in self.model_conf['label']
    ]
    total_metrics = [
        BucketAUCUtil(name=conf['input_name'], batch_size=batch_size)
        for conf in self.model_conf['label']
    ]
    step = 0
    while True:
      try:
        if self.profile:
          self.tb_callback.on_train_batch_begin(step)
        start_time = time.time()
        batch = train_itor.getbatch()
        input_dic = self.get_batch_input(batch, local_db)
        self.adam_restart = False
        output_dic = train_step(self.model, input_dic, ctx)
        if output_dic['nan_flag'] > 0:
          print('nan and skip')
          local_db.push(batch['task_id'], None, None)
          continue

        # Update training metric.
        for gt, pred, loss, metric, total_metric in zip(
            output_dic['ground_truth'], output_dic['prediction'],
            output_dic['losses'], metrics, total_metrics):
          gt = gt.numpy()
          pred = pred.numpy()
          loss = loss.numpy()
          total_metric.add(loss, gt, pred)
          if self.log_step_num < 1 or step % self.log_step_num != 0:
            continue
          #print('pred', pred[:20])

          metric.add(loss, gt, pred)
          print(f'{metric.name} {metric.calc_str()}')
          metric.reset()

        ps_grad = output_dic['ps_grad']
        local_db.push(batch['task_id'], ps_grad.numpy(), batch['payload'])
        elapse = time.time() - start_time
        if self.log_step_num >= 1 and step % self.log_step_num == 0:
          print(f'train step {step}, batch time elapse {elapse}')
          print('-' * 50)
        step += 1
        if self.profile:
          self.tb_callback.on_train_batch_end(step)
          if step > 20:
            break
        #exit()
      except EOFError:
        # save & terminate
        print('EOF, train day end.')
        break
    rst_list = []
    for metric in total_metrics:
      rst = metric.calc()
      print(f'{metric.name} {rst}')
      rst['Name'] = metric.name
      rst_list.append(rst)
    print('=' * 80)
    return rst_list

  def check_day(self, train_itor, local_db):
    ctx = self.ctx
    batch_size = ctx.batch_size
    step = 0
    embed_size = 10
    sample_data = pq.read_table('embed.parquet').to_pandas()

    while True:
      try:
        start_time = time.time()
        batch = train_itor.getbatch()
        input_dic = self.get_batch_input(batch, local_db)
        output_dic = self.model(input_dic, training=False)

        samples = sample_data[step * batch_size:step * batch_size + batch_size]
        online_embeds = samples.embed_input.to_numpy()

        # should add 'embed_input' in ouput_dic first.
        embed_input_data = output_dic['embed_input'].numpy()
        features = [feature['input_name'] for feature in self.feature_conf]
        feature_diff_data = {}
        for embed, online_embed in zip(embed_input_data, online_embeds):
          embed_off = embed.tolist()
          embed_on = online_embed.tolist()
          for idx, fea_name in enumerate(features):
            fea_embed_on = embed_on[idx * embed_size:idx * embed_size +
                                    embed_size]
            fea_embed_off = embed_off[idx * embed_size:idx * embed_size +
                                      embed_size]
            total_diff = sum([
                abs(x - y) if abs(x - y) > 1e-7 else 0
                for x, y in zip(fea_embed_on, fea_embed_off)
            ])
            if total_diff > 0 or sum(fea_embed_on) == 0:
              print(fea_name)
              print(fea_embed_on)
              print(fea_embed_off)
              feature_diff_data[fea_name] = feature_diff_data.get(fea_name,
                                                                  0) + 1
          print('-' * 60)
        print('total cnt', len(online_embeds))
        feature_diff_data = [
            [key, value] for key, value in feature_diff_data.items()
        ]
        feature_diff_data.sort(key=lambda x: x[1], reverse=True)
        print(feature_diff_data)

        # Update training metric.
        for gt, pred, conf in zip(output_dic['ground_truth'],
                                  output_dic['prediction'],
                                  self.model_conf['label']):
          gt = gt.numpy()
          pred = pred.numpy()
          print(conf['input_name'])
          thresold = 10e-4
          total = 0
          cnt = 0
          for on, off in zip(gt, pred):
            total += 1
            if abs(on - off) > thresold:
              cnt += 1
              # print(on, off)
          print('diff ratio', cnt / total)

        local_db.push(batch['task_id'])
        elapse = time.time() - start_time
        if self.log_step_num >= 1 and step % self.log_step_num == 0:
          print(f'train step {step}, batch time elapse {elapse}')
          print('-' * 50)
        step += 1
        #exit()
      except EOFError:
        # save & terminate
        print('EOF, check day end.')
        break

  def eval(self, train_itor, local_db):
    ctx = self.ctx
    batch_size = ctx.batch_size
    metrics = [
        AUCUtil(name=conf['input_name'] + '_metric')
        for conf in self.model_conf['label']
    ]
    total_metrics = [
        BucketAUCUtil(name=conf['input_name'], batch_size=batch_size)
        for conf in self.model_conf['label']
    ]
    step = 0

    while True:
      try:
        start_time = time.time()
        batch = train_itor.getbatch()
        input_dic = self.get_batch_input(batch, local_db)
        output_dic = self.model(input_dic, training=False)

        # Update training metric.
        for gt, pred, loss, metric, total_metric in zip(
            output_dic['ground_truth'], output_dic['prediction'],
            output_dic['losses'], metrics, total_metrics):
          gt = gt.numpy()
          pred = pred.numpy()
          loss = loss.numpy()
          total_metric.add(loss, gt, pred)
          if self.log_step_num < 1 or step % self.log_step_num != 0:
            continue
          metric.add(loss, gt, pred)
          print(f'{metric.name} {metric.calc_str()}')
          metric.reset()

        local_db.push(batch['task_id'])
        elapse = time.time() - start_time
        if self.log_step_num >= 1 and step % self.log_step_num == 0:
          print(f'eval step {step}, batch time elapse {elapse}')
          print('-' * 50)
        step += 1
        #exit()
      except EOFError:
        # save & terminate
        print('EOF, eval end.')
        break
    rst_list = []
    for metric in total_metrics:
      rst = metric.calc()
      print(f'{metric.name} {rst}')
      rst['Name'] = metric.name
      rst_list.append(rst)
    return rst_list

  def aucdiff_run(self, train_itor, local_db):
    ctx = self.ctx
    batch_size = ctx.batch_size
    total_metrics_list = [[
        BucketAUCUtil(name=conf['input_name'], batch_size=batch_size)
        for conf in self.model_conf['label']
    ]
                          for _ in range(ctx.origin_slot_num + 1)]
    tmask = [
        np.ones([1, self.ctx.net_input_dim], dtype=self.ctx.npval)
        for _ in range(self.ctx.origin_slot_num)
    ]  # noqa

    step = 0
    embed_offset = 0
    start_arr = []
    end_arr = []
    for mask in range(len(ctx.embed_dim)):
      pstart = embed_offset
      embed_len = self.ctx.embed_dim[mask]
      if ctx.use_acc_ctr:
        embed_len += 1
      pend = pstart + embed_len
      tmask[mask][:, pstart:pend] = 0.
      embed_offset = pend

      tmp_start = [str(pstart)]
      tmp_end = [str(pend)]
      start_arr.append('/'.join(tmp_start))
      end_arr.append('/'.join(tmp_end))

    while True:
      try:
        batch = train_itor.getbatch()
        input_dic = self.get_batch_input(batch, local_db)
        origin_mask = input_dic['test_mask']

        for mask in range(self.ctx.origin_slot_num + 1):
          if mask < ctx.origin_slot_num:
            input_dic['test_mask'] = tf.constant(tmask[mask])
          else:
            input_dic['test_mask'] = origin_mask

          output_dic = self.model(input_dic, training=False)

          for gt, pred, loss, total_metric in zip(output_dic['ground_truth'],
                                                  output_dic['prediction'],
                                                  output_dic['losses'],
                                                  total_metrics_list[mask]):
            gt = gt.numpy()
            pred = pred.numpy()
            loss = loss.numpy()
            total_metric.add(loss, gt, pred)
        step += 1
        local_db.push(batch['task_id'])
        print('aucdiff check step', step)

      except EOFError:
        data = {}
        features = [feature['input_name'] for feature in self.feature_conf]
        for mask in range(self.ctx.origin_slot_num):
          print('# Mask %s from %s to %s.' % (self.ctx.origin_slots.split(
              ',', -1)[mask], start_arr[mask], end_arr[mask]))
          feature = features[mask]
          data[feature] = {}
          for metric in total_metrics_list[mask]:
            rst = metric.calc()
            data[feature][metric.name] = rst
            print(f'{metric.name} {rst}')
        data['base'] = {}
        for metric in total_metrics_list[-1]:
          print('base metrics')
          rst = metric.calc()
          data['base'][metric.name] = rst
          print(f'{metric.name} {rst}')
        with open('aucdiff_rst.json', 'w') as target:
          json.dump(data, target)
        break

  def run(self):
    conf_overwrite(self, 'job.conf', 'job.conf.tmp')
    self.ctx = Context('job.conf.tmp')
    self.train_itor = QDataSet('job.conf.tmp', self.ctx, False)
    self.local_db = QWrapper('job.conf.tmp')
    self.b1_power = 1.
    self.b2_power = 1.
    self.adam_restart = True
    self.prepare_model()
    if self.mode.startswith('Train'):
      time_range = self.data_conf['train_days']
      data_hdfs_path = self.data_conf['data_hdfs_path'].rstrip('/') + '/'
      end_day = self.data_conf['train_end_day']
      end_time = datetime.datetime.strptime(f'{end_day} 00:00:00',
                                            '%Y-%m-%d %H:%M:%S')
      trained_days = 0
      # TODO 尝试add_history、add_progbar
      if self.profile:
        self.tb_callback.set_model(self.model)

      try:
        if self.profile:
          self.tb_callback.on_train_begin()
        model_info = {}
        for epoch in range(self.epochs):
          if self.profile:
            self.tb_callback.on_epoch_begin(epoch)
          print(f'start epoch {epoch}...')
          for i in reversed(range(time_range)):
            train_time = end_time - datetime.timedelta(days=i)
            data_path = train_time.strftime(data_hdfs_path)
            # data_date = train_time.strftime('%Y-%m-%d')
            files = get_hdfs_files(data_path)
            self.train_itor.setfiles(','.join(files))
            print('start train', train_time, data_path)
            total_metrics = self.fit_day(self.train_itor, self.local_db)
            model_info['metrics'] = total_metrics
            if self.data_conf.get('monitor'):
              for metrics in total_metrics:
                self.log_into_db(train_time, metrics)
            self.local_db.clean_nontrain(False)
            trained_days += 1
            if trained_days % 5 == 0:
              train_date = train_time.strftime('%Y-%m-%d')
              self.save_model(train_date, model_info)
          if self.profile:
            self.tb_callback.on_epoch_end(epoch)
        if self.profile:
          self.tb_callback.on_train_end()
      except Exception as e:
        logging.exception(e)
      finally:
        is_deploy = self.data_conf.get('deploy')
        model_info['version'] = int(time.mktime(train_time.timetuple()))
        train_date = train_time.strftime('%Y-%m-%d')
        model_info['date'] = train_date
        self.save_model(train_date, model_info, is_deploy=is_deploy)
        self.local_db.terminate()

    if 'Eval' in self.mode:
      data_hdfs_path = self.data_conf['data_hdfs_path'].rstrip('/') + '/'
      eval_day = self.data_conf['eval_day']
      eval_time = datetime.datetime.strptime(f'{eval_day} 00:00:00',
                                             '%Y-%m-%d %H:%M:%S')

      data_path = eval_time.strftime(data_hdfs_path)
      files = get_hdfs_files(data_path)
      # files = files[::10]
      self.train_itor.is_test = True
      self.train_itor.setfiles(','.join(files))
      print('start eval', eval_time, data_path)
      self.eval(self.train_itor, self.local_db)
      self.local_db.terminate()

    # TODO support accctr
    if self.mode == 'AucDiff':
      data_hdfs_path = self.data_conf['data_hdfs_path'].rstrip('/') + '/'
      eval_day = self.data_conf['eval_day']
      eval_time = datetime.datetime.strptime(f'{eval_day} 00:00:00',
                                             '%Y-%m-%d %H:%M:%S')

      data_path = eval_time.strftime(data_hdfs_path)
      files = get_hdfs_files(data_path)
      # files = files[::10]
      self.train_itor.is_test = True
      self.train_itor.setfiles(','.join(files))
      print('start check', eval_time, data_path)
      self.aucdiff_run(self.train_itor, self.local_db)
      self.local_db.terminate()

    # TODO support accctr
    if self.mode == 'Check':
      data_hdfs_path = self.data_conf['data_hdfs_path'].rstrip('/') + '/'
      eval_day = self.data_conf['eval_day']
      eval_time = datetime.datetime.strptime(f'{eval_day} 00:00:00',
                                             '%Y-%m-%d %H:%M:%S')

      data_path = eval_time.strftime(data_hdfs_path)
      files = get_hdfs_files(data_path)
      self.train_itor.is_test = True
      self.train_itor.setfiles(','.join(files))
      print('start check', eval_time, data_path)

      total_metrics = self.check_day(self.train_itor, self.local_db)
      self.local_db.terminate()
