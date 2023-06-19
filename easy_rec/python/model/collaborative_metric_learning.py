import tensorflow as tf

from easy_rec.python.core.metrics import metric_learning_average_precision_at_k
from easy_rec.python.core.metrics import metric_learning_recall_at_k
from easy_rec.python.layers import dnn
from easy_rec.python.layers.common_layers import highway
from easy_rec.python.loss.circle_loss import circle_loss
from easy_rec.python.loss.multi_similarity import ms_loss
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.utils.activation import gelu
from easy_rec.python.utils.proto_util import copy_obj

from easy_rec.python.protos.collaborative_metric_learning_pb2 import CoMetricLearningI2I as MetricLearningI2IConfig  # NOQA

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class CoMetricLearningI2I(EasyRecModel):

  def __init__(
      self,
      model_config,  # pipeline.model_config
      feature_configs,  # pipeline.feature_configs
      features,  # same as model_fn input
      labels=None,
      is_training=False):
    super(CoMetricLearningI2I, self).__init__(model_config, feature_configs,
                                              features, labels, is_training)
    model = self._model_config.WhichOneof('model')
    assert model == 'metric_learning', 'invalid model config: %s' % model

    self._loss_type = self._model_config.loss_type
    loss_type_name = LossType.Name(self._loss_type).lower()

    self._model_config = self._model_config.metric_learning
    assert isinstance(self._model_config, MetricLearningI2IConfig)

    model_loss = self._model_config.WhichOneof('loss').lower()
    assert model_loss == loss_type_name, 'invalid loss type: %s' % model_loss

    if self._loss_type == LossType.CIRCLE_LOSS:
      self.loss = self._model_config.circle_loss
    elif self._loss_type == LossType.MULTI_SIMILARITY_LOSS:
      self.loss = self._model_config.multi_similarity_loss
    else:
      raise ValueError('unsupported loss type: %s' %
                       LossType.Name(self._loss_type))

    if not self.has_backbone:
      self._highway_features = {}
      self._highway_num = len(self._model_config.highway)
      for _id in range(self._highway_num):
        highway_cfg = self._model_config.highway[_id]
        highway_feature, _ = self._input_layer(self._feature_dict,
                                               highway_cfg.input)
        self._highway_features[highway_cfg.input] = highway_feature

      self.input_features = []
      if self._model_config.HasField('input'):
        input_feature, _ = self._input_layer(self._feature_dict,
                                             self._model_config.input)
        self.input_features.append(input_feature)

      self.dnn = copy_obj(self._model_config.dnn)

    if self._labels is not None:
      if self._model_config.HasField('session_id'):
        self.session_ids = self._labels.pop(self._model_config.session_id)
      else:
        self.session_ids = None

      assert len(self._labels) > 0
      self.labels = list(self._labels.values())[0]

    if self._model_config.HasField('sample_id'):
      self.sample_id = self._model_config.sample_id
    else:
      self.sample_id = None

  def build_predict_graph(self):
    if self.has_backbone:
      tower_emb = self.backbone
    else:
      for _id in range(self._highway_num):
        highway_cfg = self._model_config.highway[_id]
        highway_fea = tf.layers.batch_normalization(
            self._highway_features[highway_cfg.input],
            training=self._is_training,
            trainable=True,
            name='highway_%s_bn' % highway_cfg.input)
        highway_fea = highway(
            highway_fea,
            highway_cfg.emb_size,
            activation=gelu,
            scope='highway_%s' % _id)
        print('highway_fea: ', highway_fea)
        self.input_features.append(highway_fea)

      feature = tf.concat(self.input_features, axis=1)

      num_dnn_layer = len(self.dnn.hidden_units)
      last_hidden = self.dnn.hidden_units.pop()
      dnn_net = dnn.DNN(self.dnn, self._l2_reg, 'dnn', self._is_training)
      net_output = dnn_net(feature)
      tower_emb = tf.layers.dense(
          inputs=net_output,
          units=last_hidden,
          kernel_regularizer=self._l2_reg,
          name='dnn/dnn_%d' % (num_dnn_layer - 1))

    if self._model_config.output_l2_normalized_emb:
      norm_emb = tf.nn.l2_normalize(tower_emb, axis=-1)
      self._prediction_dict['norm_emb'] = norm_emb
      self._prediction_dict['norm_embedding'] = tf.reduce_join(
          tf.as_string(norm_emb), axis=-1, separator=',')

    self._prediction_dict['float_emb'] = tower_emb
    self._prediction_dict['embedding'] = tf.reduce_join(
        tf.as_string(tower_emb), axis=-1, separator=',')
    if self.sample_id is not None and self.sample_id in self._feature_dict:
      self._prediction_dict['sample_id'] = tf.identity(
          self._feature_dict[self.sample_id])
    return self._prediction_dict

  def build_loss_graph(self):
    emb = self._prediction_dict['float_emb']
    emb_normed = self._model_config.output_l2_normalized_emb
    norm_emb = self._prediction_dict['norm_emb'] if emb_normed else emb
    if self._loss_type == LossType.CIRCLE_LOSS:
      self._loss_dict['circle_loss'] = circle_loss(
          norm_emb,
          self.labels,
          self.session_ids,
          self.loss.margin,
          self.loss.gamma,
          embed_normed=emb_normed)
    elif self._loss_type == LossType.MULTI_SIMILARITY_LOSS:
      self._loss_dict['ms_loss'] = ms_loss(
          norm_emb,
          self.labels,
          self.session_ids,
          self.loss.alpha,
          self.loss.beta,
          self.loss.lamb,
          self.loss.eps,
          embed_normed=emb_normed)
    else:
      raise ValueError('invalid loss type: %s' % LossType.Name(self._loss_type))

    return self._loss_dict

  def get_outputs(self):
    outputs = ['embedding', 'float_emb']
    if self.sample_id is not None and 'sample_id' in self._prediction_dict:
      outputs.append('sample_id')
    if self._model_config.output_l2_normalized_emb:
      outputs.append('norm_embedding')
      outputs.append('norm_emb')
    return outputs

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    recall_at_k = []
    precision_at_k = []
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'recall_at_topk':
        recall_at_k.append(metric.recall_at_topk.topk)
      elif metric.WhichOneof('metric') == 'precision_at_topk':
        precision_at_k.append(metric.precision_at_topk.topk)

    emb = self._prediction_dict['float_emb']
    if len(recall_at_k) > 0:
      metric_dict.update(
          metric_learning_recall_at_k(recall_at_k, emb, self.labels,
                                      self.session_ids))
    if len(precision_at_k) > 0:
      metric_dict.update(
          metric_learning_average_precision_at_k(precision_at_k, emb,
                                                 self.labels, self.session_ids))
    return metric_dict
