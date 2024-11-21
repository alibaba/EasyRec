# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import tensorflow as tf

from easy_rec.python.builders import loss_builder
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.protos.simi_pb2 import Similarity

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
losses = tf.losses


class MatchModel(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MatchModel, self).__init__(model_config, feature_configs, features,
                                     labels, is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class

    if self._loss_type == LossType.CLASSIFICATION:
      assert self._num_class == 1

    if self._loss_type in [LossType.CLASSIFICATION, LossType.L2_LOSS]:
      self._is_point_wise = True
      logging.info('Use point wise dssm.')
    else:
      self._is_point_wise = False
      logging.info('Use list wise dssm.')

    cls_mem = self._model_config.WhichOneof('model')
    sub_model_config = getattr(self._model_config, cls_mem)

    self._item_ids = None
    assert sub_model_config is not None, 'sub_model_config undefined: model_cls = %s' % cls_mem
    if getattr(sub_model_config, 'item_id', '') != '':
      logging.info('item_id feature is: %s' % sub_model_config.item_id)
      self._item_ids = features[sub_model_config.item_id]

  def _mask_in_batch(self, logits):
    batch_size = tf.shape(logits)[0]
    if getattr(self._model_config, 'ignore_in_batch_neg_sam', False):
      in_batch = logits[:, :batch_size] - (
          1 - tf.diag(tf.ones([batch_size], dtype=tf.float32))) * 1e32
      return tf.concat([in_batch, logits[:, batch_size:]], axis=1)
    else:
      if self._item_ids is not None:
        mask_in_batch_neg = tf.to_float(
            tf.equal(self._item_ids[None, :batch_size],
                     self._item_ids[:batch_size, None])) - tf.diag(
                         tf.ones([batch_size], dtype=tf.float32))
        tf.summary.scalar('in_batch_neg_conflict',
                          tf.reduce_sum(mask_in_batch_neg))
        return tf.concat([
            logits[:, :batch_size] - mask_in_batch_neg * 1e32,
            logits[:, batch_size:]],
            axis=1)  # yapf: disable
      else:
        return logits

  def _list_wise_sim(self, user_emb, item_emb):
    batch_size = tf.shape(user_emb)[0]
    hard_neg_indices = self._feature_dict.get('hard_neg_indices', None)

    if hard_neg_indices is not None:
      logging.info('With hard negative examples')
      noclk_size = tf.shape(hard_neg_indices)[0]
      # pos_item_emb, neg_item_emb, hard_neg_item_emb = tf.split(
      #     item_emb, [batch_size, -1, noclk_size], axis=0)
      simple_item_emb, hard_neg_item_emb = tf.split(
          item_emb, [-1, noclk_size], axis=0)
    else:
      # pos_item_emb = item_emb[:batch_size]
      # neg_item_emb = item_emb[batch_size:]
      simple_item_emb = item_emb

    # pos_user_item_sim = tf.reduce_sum(
    #     tf.multiply(user_emb, pos_item_emb), axis=1, keep_dims=True)
    # neg_user_item_sim = tf.matmul(user_emb, tf.transpose(neg_item_emb))
    # simple_user_item_sim = tf.matmul(user_emb, tf.transpose(simple_item_emb))

    _mode = os.environ['tf.estimator.mode']
    if _mode == tf.estimator.ModeKeys.PREDICT:
      simple_user_item_sim = tf.reduce_sum(
          tf.multiply(user_emb, simple_item_emb), axis=1, keep_dims=True)
    else:
      simple_user_item_sim = tf.matmul(user_emb, tf.transpose(simple_item_emb))

    if hard_neg_indices is None:
      return simple_user_item_sim
    else:
      user_emb_expand = tf.gather(user_emb, hard_neg_indices[:, 0])
      hard_neg_user_item_sim = tf.reduce_sum(
          tf.multiply(user_emb_expand, hard_neg_item_emb), axis=1)
      max_num_neg = tf.reduce_max(hard_neg_indices[:, 1]) + 1
      hard_neg_shape = tf.stack([tf.to_int64(batch_size), max_num_neg])
      hard_neg_sim = tf.scatter_nd(hard_neg_indices, hard_neg_user_item_sim,
                                   hard_neg_shape)
      hard_neg_mask = tf.scatter_nd(
          hard_neg_indices,
          tf.ones_like(hard_neg_user_item_sim, dtype=tf.float32),
          shape=hard_neg_shape)
      # set tail positions to -1e32, so that after exp(x), will be zero
      hard_neg_user_item_sim = hard_neg_sim - (1 - hard_neg_mask) * 1e32

      # user_item_sim = [pos_user_item_sim, neg_user_item_sim]
      # if hard_neg_indices is not None:
      #   user_item_sim.append(hard_neg_user_item_sim)
      # return tf.concat(user_item_sim, axis=1)

      return tf.concat([simple_user_item_sim, hard_neg_user_item_sim], axis=1)

  def _point_wise_sim(self, user_emb, item_emb):
    user_item_sim = tf.reduce_sum(
        tf.multiply(user_emb, item_emb), axis=1, keep_dims=True)
    return user_item_sim

  def sim(self, user_emb, item_emb):
    # Name the outputs of the user tower and the item tower, i.e. the inputs of the
    # simularity operation.
    # Explicit names of these nodes are necessary for some online recall systems like
    # BasicEngine to split up the predicting graph into different clusters.
    user_emb = tf.identity(user_emb, 'user_tower_emb')
    item_emb = tf.identity(item_emb, 'item_tower_emb')

    if self._is_point_wise:
      return self._point_wise_sim(user_emb, item_emb)
    else:
      return self._list_wise_sim(user_emb, item_emb)

  def norm(self, fea):
    fea_norm = tf.nn.l2_normalize(fea, axis=-1)
    return fea_norm

  def build_predict_graph(self):
    if not self.has_backbone:
      raise NotImplementedError(
          'method `build_predict_graph` must be implemented when you donot use backbone network'
      )
    model = self._model_config.WhichOneof('model')
    assert model == 'model_params', '`model_params` must be configured'
    model_params = self._model_config.model_params
    for out in model_params.outputs:
      self._outputs.append(out)

    output = self.backbone

    user_tower_emb = output[model_params.user_tower_idx_in_output]
    item_tower_emb = output[model_params.item_tower_idx_in_output]

    if model_params.simi_func == Similarity.COSINE:
      user_tower_emb = self.norm(user_tower_emb)
      item_tower_emb = self.norm(item_tower_emb)
      temperature = model_params.temperature
    else:
      temperature = 1.0

    user_item_sim = self.sim(user_tower_emb, item_tower_emb) / temperature

    if model_params.scale_simi:
      sim_w = tf.get_variable(
          'sim_w',
          dtype=tf.float32,
          shape=(1),
          initializer=tf.ones_initializer())
      sim_b = tf.get_variable(
          'sim_b',
          dtype=tf.float32,
          shape=(1),
          initializer=tf.zeros_initializer())
      y_pred = user_item_sim * tf.abs(sim_w) + sim_b
    else:
      y_pred = user_item_sim

    if self._is_point_wise:
      y_pred = tf.reshape(y_pred, [-1])

    if self._loss_type == LossType.CLASSIFICATION:
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.sigmoid(y_pred)
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      y_pred = self._mask_in_batch(y_pred)
      self._prediction_dict['logits'] = y_pred
      self._prediction_dict['probs'] = tf.nn.softmax(y_pred)
    else:
      self._prediction_dict['y'] = y_pred

    self._prediction_dict['user_tower_emb'] = user_tower_emb
    self._prediction_dict['item_tower_emb'] = item_tower_emb
    self._prediction_dict['user_emb'] = tf.reduce_join(
        tf.as_string(user_tower_emb), axis=-1, separator=',')
    self._prediction_dict['item_emb'] = tf.reduce_join(
        tf.as_string(item_tower_emb), axis=-1, separator=',')

    return self._prediction_dict

  def build_loss_graph(self):
    if self._is_point_wise:
      return self._build_point_wise_loss_graph()
    else:
      return self._build_list_wise_loss_graph()

  def _build_list_wise_loss_graph(self):
    if self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      batch_size = tf.shape(self._prediction_dict['probs'])[0]
      indices = tf.range(batch_size)
      indices = tf.concat([indices[:, None], indices[:, None]], axis=1)
      hit_prob = tf.gather_nd(
          self._prediction_dict['probs'][:batch_size, :batch_size], indices)
      self._loss_dict['cross_entropy_loss'] = -tf.reduce_mean(
          tf.log(hit_prob + 1e-12) * tf.squeeze(self._sample_weight))
      logging.info('softmax cross entropy loss is used')

      user_features = self._prediction_dict['user_tower_emb']
      pos_item_emb = self._prediction_dict['item_tower_emb'][:batch_size]
      pos_simi = tf.reduce_sum(user_features * pos_item_emb, axis=1)
      # if pos_simi < 0, produce loss
      reg_pos_loss = tf.nn.relu(-pos_simi)
      self._loss_dict['reg_pos_loss'] = tf.reduce_mean(reg_pos_loss)

      # the AMM loss for DAT model
      if all([
          k in self._prediction_dict.keys() for k in
          ['augmented_p_u', 'augmented_p_i', 'augmented_a_u', 'augmented_a_i']
      ]):
        self._loss_dict['amm_loss_u'] = tf.reduce_mean(
            tf.square(self._prediction_dict['augmented_a_u'] -
                      self._prediction_dict['augmented_p_i'][:batch_size]))
        self._loss_dict['amm_loss_i'] = tf.reduce_mean(
            tf.square(self._prediction_dict['augmented_a_i'][:batch_size] -
                      self._prediction_dict['augmented_p_u']))

    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
    return self._loss_dict

  def _build_point_wise_loss_graph(self):
    label = list(self._labels.values())[0]
    if self._loss_type == LossType.CLASSIFICATION:
      pred = self._prediction_dict['logits']
      loss_name = 'cross_entropy_loss'
    elif self._loss_type == LossType.L2_LOSS:
      pred = self._prediction_dict['y']
      loss_name = 'l2_loss'
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))

    kwargs = {'loss_name': loss_name}
    self._loss_dict[loss_name] = loss_builder.build(
        self._loss_type,
        label=label,
        pred=pred,
        loss_weight=self._sample_weight,
        **kwargs)

    # build kd loss
    kd_loss_dict = loss_builder.build_kd_loss(self.kd, self._prediction_dict,
                                              self._labels, self._feature_dict)
    self._loss_dict.update(kd_loss_dict)
    return self._loss_dict

  def build_metric_graph(self, eval_config):
    if self._is_point_wise:
      return self._build_point_wise_metric_graph(eval_config)
    else:
      return self._build_list_wise_metric_graph(eval_config)

  def _build_list_wise_metric_graph(self, eval_config):
    from easy_rec.python.core.easyrec_metrics import metrics_tf
    logits = self._prediction_dict['logits']
    # label = tf.zeros_like(logits[:, :1], dtype=tf.int64)
    batch_size = tf.shape(logits)[0]
    label = tf.cast(tf.range(batch_size), tf.int64)

    indices = tf.range(batch_size)
    indices = tf.concat([indices[:, None], indices[:, None]], axis=1)
    pos_item_sim = tf.gather_nd(logits[:batch_size, :batch_size], indices)
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'recall_at_topk':
        metric_dict['recall@%d' %
                    metric.recall_at_topk.topk] = metrics_tf.recall_at_k(
                        label, logits, metric.recall_at_topk.topk)

        logits_v2 = tf.concat([pos_item_sim[:, None], logits[:, batch_size:]],
                              axis=1)
        labels_v2 = tf.zeros_like(logits_v2[:, :1], dtype=tf.int64)
        metric_dict['recall_neg_sam@%d' %
                    metric.recall_at_topk.topk] = metrics_tf.recall_at_k(
                        labels_v2, logits_v2, metric.recall_at_topk.topk)

        metric_dict['recall_in_batch@%d' %
                    metric.recall_at_topk.topk] = metrics_tf.recall_at_k(
                        label, logits[:, :batch_size],
                        metric.recall_at_topk.topk)
      else:
        raise ValueError('invalid metric type: %s' % str(metric))
    return metric_dict

  def _build_point_wise_metric_graph(self, eval_config):
    from easy_rec.python.core.easyrec_metrics import metrics_tf
    metric_dict = {}
    label = list(self._labels.values())[0]
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._loss_type == LossType.CLASSIFICATION
        metric_dict['auc'] = metrics_tf.auc(label,
                                            self._prediction_dict['probs'])
      elif metric.WhichOneof('metric') == 'mean_absolute_error':
        assert self._loss_type == LossType.L2_LOSS
        metric_dict['mean_absolute_error'] = metrics_tf.mean_absolute_error(
            tf.to_float(label), self._prediction_dict['y'])
      else:
        raise ValueError('invalid metric type: %s' % str(metric))
    return metric_dict

  def get_outputs(self):
    if not self.has_backbone:
      raise NotImplementedError(
          'could not call get_outputs on abstract class MatchModel')
    if self._loss_type == LossType.CLASSIFICATION:
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_tower_emb',
          'item_tower_emb'
      ]
    elif self._loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
      self._prediction_dict['logits'] = tf.squeeze(
          self._prediction_dict['logits'], axis=-1)
      self._prediction_dict['probs'] = tf.nn.sigmoid(
          self._prediction_dict['logits'])
      return [
          'logits', 'probs', 'user_emb', 'item_emb', 'user_tower_emb',
          'item_tower_emb'
      ]
    elif self._loss_type == LossType.L2_LOSS:
      return ['y', 'user_emb', 'item_emb', 'user_tower_emb', 'item_tower_emb']
    else:
      raise ValueError('invalid loss type: %s' % str(self._loss_type))
