# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf
from tensorflow.python.ops import math_ops

from easy_rec.python.builders import loss_builder
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType

from easy_rec.python.loss.zero_inflated_lognormal import zero_inflated_lognormal_pred  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class RankModel(EasyRecModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(RankModel, self).__init__(model_config, feature_configs, features,
                                    labels, is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class
    self._losses = self._model_config.losses
    if self._labels is not None:
      if model_config.HasField('label_name'):
        self._label_name = model_config.label_name
      else:
        self._label_name = list(self._labels.keys())[0]
    self._outputs = []

  def build_predict_graph(self):
    if not self.has_backbone:
      raise NotImplementedError(
          'method `build_predict_graph` must be implemented when backbone network do not exits'
      )
    model = self._model_config.WhichOneof('model')
    assert model == 'model_params', '`model_params` must be configured'
    config = self._model_config.model_params
    for out in config.outputs:
      self._outputs.append(out)

    output = self.backbone
    if int(output.shape[-1]) != self._num_class:
      logging.info('add head logits layer for rank model')
      output = tf.layers.dense(output, self._num_class, name='output')
    self._add_to_prediction_dict(output)
    return self._prediction_dict

  def _output_to_prediction_impl(self,
                                 output,
                                 loss_type,
                                 num_class=1,
                                 suffix=''):
    prediction_dict = {}
    binary_loss_type = {
        LossType.F1_REWEIGHTED_LOSS, LossType.PAIR_WISE_LOSS,
        LossType.BINARY_FOCAL_LOSS, LossType.PAIRWISE_FOCAL_LOSS,
        LossType.LISTWISE_RANK_LOSS, LossType.PAIRWISE_HINGE_LOSS,
        LossType.PAIRWISE_LOGISTIC_LOSS, LossType.BINARY_CROSS_ENTROPY_LOSS,
        LossType.LISTWISE_DISTILL_LOSS
    }
    if loss_type in binary_loss_type:
      assert num_class == 1, 'num_class must be 1 when loss type is %s' % loss_type.name
      output = tf.squeeze(output, axis=1)
      probs = tf.sigmoid(output)
      tf.summary.scalar('prediction/probs', tf.reduce_mean(probs))
      prediction_dict['logits' + suffix] = output
      prediction_dict['probs' + suffix] = probs
    elif loss_type == LossType.JRC_LOSS:
      assert num_class == 2, 'num_class must be 2 when loss type is JRC_LOSS'
      probs = tf.nn.softmax(output, axis=1)
      tf.summary.scalar('prediction/probs', tf.reduce_mean(probs[:, 1]))
      prediction_dict['logits' + suffix] = output
      prediction_dict['pos_logits' + suffix] = output[:, 1]
      prediction_dict['probs' + suffix] = probs[:, 1]
    elif loss_type == LossType.ZILN_LOSS:
      assert num_class == 3, 'num_class must be 3 when loss type is ZILN_LOSS'
      probs, preds = zero_inflated_lognormal_pred(output)
      tf.summary.scalar('prediction/probs', tf.reduce_mean(probs))
      tf.summary.scalar('prediction/y', tf.reduce_mean(preds))
      prediction_dict['logits' + suffix] = output
      prediction_dict['probs' + suffix] = probs
      prediction_dict['y' + suffix] = preds
    elif loss_type == LossType.CLASSIFICATION:
      if num_class == 1:
        output = tf.squeeze(output, axis=1)
        probs = tf.sigmoid(output)
        tf.summary.scalar('prediction/probs', tf.reduce_mean(probs))
        prediction_dict['logits' + suffix] = output
        prediction_dict['probs' + suffix] = probs
      else:
        probs = tf.nn.softmax(output, axis=1)
        prediction_dict['logits' + suffix] = output
        prediction_dict['logits' + suffix + '_1'] = output[:, 1]
        prediction_dict['probs' + suffix] = probs
        prediction_dict['probs' + suffix + '_1'] = probs[:, 1]
        prediction_dict['logits' + suffix + '_y'] = math_ops.reduce_max(
            output, axis=1)
        prediction_dict['probs' + suffix + '_y'] = math_ops.reduce_max(
            probs, axis=1)
        prediction_dict['y' + suffix] = tf.argmax(output, axis=1)
    elif loss_type == LossType.L2_LOSS:
      output = tf.squeeze(output, axis=1)
      prediction_dict['y' + suffix] = output
    elif loss_type == LossType.SIGMOID_L2_LOSS:
      output = tf.squeeze(output, axis=1)
      prediction_dict['y' + suffix] = tf.sigmoid(output)
    return prediction_dict

  def _add_to_prediction_dict(self, output):
    if len(self._losses) == 0:
      prediction_dict = self._output_to_prediction_impl(
          output, loss_type=self._loss_type, num_class=self._num_class)
      self._prediction_dict.update(prediction_dict)
    else:
      for loss in self._losses:
        prediction_dict = self._output_to_prediction_impl(
            output, loss_type=loss.loss_type, num_class=self._num_class)
        self._prediction_dict.update(prediction_dict)

  def build_rtp_output_dict(self):
    """Forward tensor as `rank_predict`, which is a special node for RTP."""
    outputs = {}
    outputs.update(super(RankModel, self).build_rtp_output_dict())
    rank_predict = None
    try:
      op = tf.get_default_graph().get_operation_by_name('rank_predict')
      if len(op.outputs) != 1:
        raise ValueError(
            ('failed to build RTP rank_predict output: op {}[{}] has output ' +
             'size {}, however 1 is expected.').format(op.name, op.type,
                                                       len(op.outputs)))
      rank_predict = op.outputs[0]
    except KeyError:
      forwarded = None
      loss_types = {self._loss_type}
      if len(self._losses) > 0:
        loss_types = {loss.loss_type for loss in self._losses}
      binary_loss_set = {
          LossType.CLASSIFICATION, LossType.F1_REWEIGHTED_LOSS,
          LossType.PAIR_WISE_LOSS, LossType.BINARY_FOCAL_LOSS,
          LossType.PAIRWISE_FOCAL_LOSS, LossType.PAIRWISE_LOGISTIC_LOSS,
          LossType.JRC_LOSS, LossType.LISTWISE_DISTILL_LOSS,
          LossType.LISTWISE_RANK_LOSS
      }
      if loss_types & binary_loss_set:
        if 'probs' in self._prediction_dict:
          forwarded = self._prediction_dict['probs']
        else:
          raise ValueError(
              'failed to build RTP rank_predict output: classification model ' +
              "expect 'probs' prediction, which is not found. Please check if" +
              ' build_predict_graph() is called.')
      elif loss_types & {
          LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS, LossType.ZILN_LOSS
      }:
        if 'y' in self._prediction_dict:
          forwarded = self._prediction_dict['y']
        else:
          raise ValueError(
              'failed to build RTP rank_predict output: regression model expect'
              +
              "'y' prediction, which is not found. Please check if build_predic"
              + 't_graph() is called.')
      else:
        logging.warning(
            'failed to build RTP rank_predict: unsupported loss type {}'.format(
                loss_types))
      if forwarded is not None:
        rank_predict = tf.identity(forwarded, name='rank_predict')
    if rank_predict is not None:
      outputs['rank_predict'] = rank_predict
    return outputs

  def _build_loss_impl(self,
                       loss_type,
                       label_name,
                       loss_weight=1.0,
                       num_class=1,
                       suffix='',
                       loss_name='',
                       loss_param=None):
    loss_dict = {}
    binary_loss_type = {
        LossType.F1_REWEIGHTED_LOSS, LossType.PAIR_WISE_LOSS,
        LossType.BINARY_FOCAL_LOSS, LossType.PAIRWISE_FOCAL_LOSS,
        LossType.LISTWISE_RANK_LOSS, LossType.PAIRWISE_HINGE_LOSS,
        LossType.PAIRWISE_LOGISTIC_LOSS, LossType.JRC_LOSS,
        LossType.LISTWISE_DISTILL_LOSS, LossType.ZILN_LOSS
    }
    if loss_type in {
        LossType.CLASSIFICATION, LossType.BINARY_CROSS_ENTROPY_LOSS
    }:
      loss_name = loss_name if loss_name else 'cross_entropy_loss' + suffix
      pred = self._prediction_dict['logits' + suffix]
    elif loss_type in binary_loss_type:
      if not loss_name:
        loss_name = LossType.Name(loss_type).lower() + suffix
      else:
        loss_name = loss_name + suffix
      pred = self._prediction_dict['logits' + suffix]
    elif loss_type in [LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS]:
      loss_name = loss_name if loss_name else 'l2_loss' + suffix
      pred = self._prediction_dict['y' + suffix]
    else:
      raise ValueError('invalid loss type: %s' % LossType.Name(loss_type))

    tf.summary.scalar('labels/%s' % label_name,
                      tf.reduce_mean(tf.to_float(self._labels[label_name])))
    kwargs = {'loss_name': loss_name}
    if loss_param is not None:
      if hasattr(loss_param, 'session_name'):
        kwargs['session_ids'] = self._feature_dict[loss_param.session_name]
    loss_dict[loss_name] = loss_builder.build(
        loss_type,
        self._labels[label_name],
        pred,
        loss_weight,
        num_class,
        loss_param=loss_param,
        **kwargs)
    return loss_dict

  def build_loss_graph(self):
    loss_dict = {}
    with tf.name_scope('loss'):
      if len(self._losses) == 0:
        loss_dict = self._build_loss_impl(
            self._loss_type,
            label_name=self._label_name,
            loss_weight=self._sample_weight,
            num_class=self._num_class)
      else:
        strategy = self._base_model_config.loss_weight_strategy
        loss_weight = [1.0]
        if strategy == self._base_model_config.Random and len(self._losses) > 1:
          weights = tf.random_normal([len(self._losses)])
          loss_weight = tf.nn.softmax(weights)
        for i, loss in enumerate(self._losses):
          loss_param = loss.WhichOneof('loss_param')
          if loss_param is not None:
            loss_param = getattr(loss, loss_param)
          loss_ops = self._build_loss_impl(
              loss.loss_type,
              label_name=self._label_name,
              loss_weight=self._sample_weight,
              num_class=self._num_class,
              loss_name=loss.loss_name,
              loss_param=loss_param)
          for loss_name, loss_value in loss_ops.items():
            if strategy == self._base_model_config.Fixed:
              loss_dict[loss_name] = loss_value * loss.weight
            elif strategy == self._base_model_config.Uncertainty:
              if loss.learn_loss_weight:
                uncertainty = tf.Variable(
                    0, name='%s_loss_weight' % loss_name, dtype=tf.float32)
                tf.summary.scalar('%s_uncertainty' % loss_name, uncertainty)
                if loss.loss_type in {
                    LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS
                }:
                  loss_dict[loss_name] = 0.5 * tf.exp(
                      -uncertainty) * loss_value + 0.5 * uncertainty
                else:
                  loss_dict[loss_name] = tf.exp(
                      -uncertainty) * loss_value + 0.5 * uncertainty
              else:
                loss_dict[loss_name] = loss_value * loss.weight
            elif strategy == self._base_model_config.Random:
              loss_dict[loss_name] = loss_value * loss_weight[i]
            else:
              raise ValueError('Unsupported loss weight strategy: ' +
                               strategy.Name)
      self._loss_dict.update(loss_dict)
      # build kd loss
      kd_loss_dict = loss_builder.build_kd_loss(self.kd, self._prediction_dict,
                                                self._labels,
                                                self._feature_dict)
      self._loss_dict.update(kd_loss_dict)
    return self._loss_dict

  def _build_metric_impl(self,
                         metric,
                         loss_type,
                         label_name,
                         num_class=1,
                         suffix=''):
    if not isinstance(loss_type, set):
      loss_type = {loss_type}
    from easy_rec.python.core.easyrec_metrics import metrics_tf
    from easy_rec.python.core import metrics as metrics_lib
    binary_loss_set = {
        LossType.CLASSIFICATION, LossType.F1_REWEIGHTED_LOSS,
        LossType.PAIR_WISE_LOSS, LossType.BINARY_FOCAL_LOSS,
        LossType.PAIRWISE_FOCAL_LOSS, LossType.PAIRWISE_LOGISTIC_LOSS,
        LossType.JRC_LOSS, LossType.LISTWISE_DISTILL_LOSS,
        LossType.LISTWISE_RANK_LOSS, LossType.ZILN_LOSS
    }
    metric_dict = {}
    if metric.WhichOneof('metric') == 'auc':
      assert loss_type & binary_loss_set
      if num_class == 1 or loss_type & {LossType.JRC_LOSS, LossType.ZILN_LOSS}:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['auc' + suffix] = metrics_tf.auc(
            label,
            self._prediction_dict['probs' + suffix],
            num_thresholds=metric.auc.num_thresholds)
      elif num_class == 2:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['auc' + suffix] = metrics_tf.auc(
            label,
            self._prediction_dict['probs' + suffix][:, 1],
            num_thresholds=metric.auc.num_thresholds)
      else:
        raise ValueError('Wrong class number')
    elif metric.WhichOneof('metric') == 'gauc':
      assert loss_type & binary_loss_set
      if num_class == 1 or loss_type & {LossType.JRC_LOSS, LossType.ZILN_LOSS}:
        label = tf.to_int64(self._labels[label_name])
        uids = self._feature_dict[metric.gauc.uid_field]
        if isinstance(uids, tf.sparse.SparseTensor):
          uids = tf.sparse_to_dense(
              uids.indices, uids.dense_shape, uids.values, default_value='')
          uids = tf.reshape(uids, [-1])
        metric_dict['gauc' + suffix] = metrics_lib.gauc(
            label,
            self._prediction_dict['probs' + suffix],
            uids=uids,
            reduction=metric.gauc.reduction)
      elif num_class == 2:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['gauc' + suffix] = metrics_lib.gauc(
            label,
            self._prediction_dict['probs' + suffix][:, 1],
            uids=self._feature_dict[metric.gauc.uid_field],
            reduction=metric.gauc.reduction)
      else:
        raise ValueError('Wrong class number')
    elif metric.WhichOneof('metric') == 'session_auc':
      assert loss_type & binary_loss_set
      if num_class == 1 or loss_type & {LossType.JRC_LOSS, LossType.ZILN_LOSS}:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['session_auc' + suffix] = metrics_lib.session_auc(
            label,
            self._prediction_dict['probs' + suffix],
            session_ids=self._feature_dict[metric.session_auc.session_id_field],
            reduction=metric.session_auc.reduction)
      elif num_class == 2:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['session_auc' + suffix] = metrics_lib.session_auc(
            label,
            self._prediction_dict['probs' + suffix][:, 1],
            session_ids=self._feature_dict[metric.session_auc.session_id_field],
            reduction=metric.session_auc.reduction)
      else:
        raise ValueError('Wrong class number')
    elif metric.WhichOneof('metric') == 'max_f1':
      assert loss_type & binary_loss_set
      if num_class == 1 or loss_type & {LossType.JRC_LOSS, LossType.ZILN_LOSS}:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['max_f1' + suffix] = metrics_lib.max_f1(
            label, self._prediction_dict['logits' + suffix])
      elif num_class == 2:
        label = tf.to_int64(self._labels[label_name])
        metric_dict['max_f1' + suffix] = metrics_lib.max_f1(
            label, self._prediction_dict['logits' + suffix][:, 1])
      else:
        raise ValueError('Wrong class number')
    elif metric.WhichOneof('metric') == 'recall_at_topk':
      assert loss_type & binary_loss_set
      assert num_class > 1
      label = tf.to_int64(self._labels[label_name])
      metric_dict['recall_at_topk' + suffix] = metrics_tf.recall_at_k(
          label, self._prediction_dict['logits' + suffix],
          metric.recall_at_topk.topk)
    elif metric.WhichOneof('metric') == 'mean_absolute_error':
      label = tf.to_float(self._labels[label_name])
      if loss_type & {
          LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS, LossType.ZILN_LOSS
      }:
        metric_dict['mean_absolute_error' +
                    suffix] = metrics_tf.mean_absolute_error(
                        label, self._prediction_dict['y' + suffix])
      elif loss_type & {LossType.CLASSIFICATION} and num_class == 1:
        metric_dict['mean_absolute_error' +
                    suffix] = metrics_tf.mean_absolute_error(
                        label, self._prediction_dict['probs' + suffix])
      else:
        assert False, 'mean_absolute_error is not supported for this model'
    elif metric.WhichOneof('metric') == 'mean_squared_error':
      label = tf.to_float(self._labels[label_name])
      if loss_type & {
          LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS, LossType.ZILN_LOSS
      }:
        metric_dict['mean_squared_error' +
                    suffix] = metrics_tf.mean_squared_error(
                        label, self._prediction_dict['y' + suffix])
      elif num_class == 1 and loss_type & binary_loss_set:
        metric_dict['mean_squared_error' +
                    suffix] = metrics_tf.mean_squared_error(
                        label, self._prediction_dict['probs' + suffix])
      else:
        assert False, 'mean_squared_error is not supported for this model'
    elif metric.WhichOneof('metric') == 'root_mean_squared_error':
      label = tf.to_float(self._labels[label_name])
      if loss_type & {
          LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS, LossType.ZILN_LOSS
      }:
        metric_dict['root_mean_squared_error' +
                    suffix] = metrics_tf.root_mean_squared_error(
                        label, self._prediction_dict['y' + suffix])
      elif loss_type & {LossType.CLASSIFICATION} and num_class == 1:
        metric_dict['root_mean_squared_error' +
                    suffix] = metrics_tf.root_mean_squared_error(
                        label, self._prediction_dict['probs' + suffix])
      else:
        assert False, 'root_mean_squared_error is not supported for this model'
    elif metric.WhichOneof('metric') == 'accuracy':
      assert loss_type & {LossType.CLASSIFICATION}
      assert num_class > 1
      label = tf.to_int64(self._labels[label_name])
      metric_dict['accuracy' + suffix] = metrics_tf.accuracy(
          label, self._prediction_dict['y' + suffix])
    return metric_dict

  def build_metric_graph(self, eval_config):
    loss_types = {self._loss_type}
    if len(self._losses) > 0:
      loss_types = {loss.loss_type for loss in self._losses}
    for metric in eval_config.metrics_set:
      self._metric_dict.update(
          self._build_metric_impl(
              metric,
              loss_type=loss_types,
              label_name=self._label_name,
              num_class=self._num_class))
    return self._metric_dict

  def _get_outputs_impl(self, loss_type, num_class=1, suffix=''):
    binary_loss_set = {
        LossType.F1_REWEIGHTED_LOSS, LossType.PAIR_WISE_LOSS,
        LossType.BINARY_FOCAL_LOSS, LossType.PAIRWISE_FOCAL_LOSS,
        LossType.LISTWISE_RANK_LOSS, LossType.PAIRWISE_HINGE_LOSS,
        LossType.PAIRWISE_LOGISTIC_LOSS, LossType.LISTWISE_DISTILL_LOSS
    }
    if loss_type in binary_loss_set:
      return ['probs' + suffix, 'logits' + suffix]
    if loss_type == LossType.JRC_LOSS:
      return ['probs' + suffix, 'pos_logits' + suffix]
    if loss_type == LossType.ZILN_LOSS:
      return ['probs' + suffix, 'y' + suffix, 'logits' + suffix]
    if loss_type == LossType.CLASSIFICATION:
      if num_class == 1:
        return ['probs' + suffix, 'logits' + suffix]
      else:
        return [
            'y' + suffix, 'probs' + suffix, 'logits' + suffix,
            'probs' + suffix + '_y', 'logits' + suffix + '_y',
            'probs' + suffix + '_1', 'logits' + suffix + '_1'
        ]
    elif loss_type in [LossType.L2_LOSS, LossType.SIGMOID_L2_LOSS]:
      return ['y' + suffix]
    else:
      raise ValueError('invalid loss type: %s' % LossType.Name(loss_type))

  def get_outputs(self):
    if len(self._losses) == 0:
      outputs = self._get_outputs_impl(self._loss_type, self._num_class)
      if self._outputs:
        outputs.extend(self._outputs)
      return list(set(outputs))

    all_outputs = []
    if self._outputs:
      all_outputs.extend(self._outputs)
    for loss in self._losses:
      outputs = self._get_outputs_impl(loss.loss_type, self._num_class)
      all_outputs.extend(outputs)
    return list(set(all_outputs))
