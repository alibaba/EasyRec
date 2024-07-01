# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import re
from abc import abstractmethod

import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile

from easy_rec.python.compat import regularizers
from easy_rec.python.layers import input_layer
from easy_rec.python.layers.backbone import Backbone
from easy_rec.python.utils import constant
from easy_rec.python.utils import estimator_utils
from easy_rec.python.utils import restore_filter
from easy_rec.python.utils.load_class import get_register_class_meta

try:
  import horovod.tensorflow as hvd
  from sparse_operation_kit.experiment import raw_ops as dynamic_variable_ops
  from sparse_operation_kit import experiment as sok
except Exception:
  dynamic_variable_ops = None
  sok = None

try:
  from tensorflow.python.framework.load_library import load_op_library
  import easy_rec
  load_embed_lib_path = os.path.join(easy_rec.ops_dir, 'libload_embed.so')
  load_embed_lib = load_op_library(load_embed_lib_path)
except Exception as ex:
  logging.warning('load libload_embed.so failed: %s' % str(ex))
  load_embed_lib = None

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

_EASY_REC_MODEL_CLASS_MAP = {}
_meta_type = get_register_class_meta(
    _EASY_REC_MODEL_CLASS_MAP, have_abstract_class=True)


class EasyRecModel(six.with_metaclass(_meta_type, object)):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    self._base_model_config = model_config
    self._model_config = model_config
    self._is_training = is_training
    self._is_predicting = labels is None
    self._feature_dict = features

    # embedding variable parameters
    self._global_ev_params = None
    if model_config.HasField('ev_params'):
      self._global_ev_params = model_config.ev_params

    if self.embedding_regularization > 0:
      self._emb_reg = regularizers.l2_regularizer(self.embedding_regularization)
    else:
      self._emb_reg = None

    if self.l2_regularization > 0:
      self._l2_reg = regularizers.l2_regularizer(self.l2_regularization)
    else:
      self._l2_reg = None

    # only used by model with wide feature groups, e.g. WideAndDeep
    self._wide_output_dim = -1
    if self.has_backbone:
      wide_dim = Backbone.wide_embed_dim(model_config.backbone)
      if wide_dim:
        self._wide_output_dim = wide_dim
        logging.info('set `wide_output_dim` to %d' % wide_dim)

    self._feature_configs = feature_configs
    self.build_input_layer(model_config, feature_configs)

    self._labels = labels
    self._prediction_dict = {}
    self._loss_dict = {}
    self._metric_dict = {}

    # add sample weight from inputs
    self._sample_weight = 1.0
    if constant.SAMPLE_WEIGHT in features:
      self._sample_weight = features[constant.SAMPLE_WEIGHT]

    self._backbone_output = None
    self._backbone_net = self.build_backbone_network()

  def build_backbone_network(self):
    if self.has_backbone:
      return Backbone(
          self._base_model_config.backbone,
          self._feature_dict,
          input_layer=self._input_layer,
          l2_reg=self._l2_reg)
    return None

  @property
  def has_backbone(self):
    return self._base_model_config.HasField('backbone')

  @property
  def backbone(self):
    if self._backbone_output:
      return self._backbone_output
    if self._backbone_net:
      kwargs = {
          'loss_dict': self._loss_dict,
          'metric_dict': self._metric_dict,
          'prediction_dict': self._prediction_dict,
          'labels': self._labels,
          constant.SAMPLE_WEIGHT: self._sample_weight
      }
      return self._backbone_net(self._is_training, **kwargs)
    return None

  @property
  def embedding_regularization(self):
    return self._base_model_config.embedding_regularization

  @property
  def kd(self):
    return self._base_model_config.kd

  @property
  def feature_groups(self):
    return self._base_model_config.feature_groups

  @property
  def l2_regularization(self):
    model_config = getattr(self._base_model_config,
                           self._base_model_config.WhichOneof('model'))
    l2_regularization = 0.0
    if hasattr(model_config, 'dense_regularization') and \
       model_config.HasField('dense_regularization'):
      # backward compatibility
      logging.warn(
          'dense_regularization is deprecated, please use l2_regularization')
      l2_regularization = model_config.dense_regularization
    elif hasattr(model_config, 'l2_regularization'):
      l2_regularization = model_config.l2_regularization
    return l2_regularization

  def build_input_layer(self, model_config, feature_configs):
    self._input_layer = input_layer.InputLayer(
        feature_configs,
        model_config.feature_groups,
        wide_output_dim=self._wide_output_dim,
        ev_params=self._global_ev_params,
        embedding_regularizer=self._emb_reg,
        kernel_regularizer=self._l2_reg,
        variational_dropout_config=model_config.variational_dropout
        if model_config.HasField('variational_dropout') else None,
        is_training=self._is_training,
        is_predicting=self._is_predicting)

  @abstractmethod
  def build_predict_graph(self):
    pass

  @abstractmethod
  def build_loss_graph(self):
    pass

  def build_metric_graph(self, eval_config):
    return self._metric_dict

  @abstractmethod
  def get_outputs(self):
    pass

  def build_output_dict(self):
    """For exporting: get standard output nodes."""
    outputs = {}
    for name in self.get_outputs():
      if name not in self._prediction_dict:
        raise KeyError(
            'output node {} not in prediction_dict, can not be exported'.format(
                name))
      outputs[name] = self._prediction_dict[name]
    return outputs

  def build_feature_output_dict(self):
    """For exporting: get output feature nodes."""
    outputs = {}
    for feature_name in self._feature_dict:
      out_name = 'feature_' + feature_name
      feature_value = self._feature_dict[feature_name]
      if isinstance(feature_value, tf.SparseTensor):
        sparse_values = feature_value.values
        if sparse_values.dtype != tf.string:
          sparse_values = tf.as_string(sparse_values)
        feature_value = tf.sparse_to_dense(feature_value.indices,
                                           feature_value.dense_shape,
                                           sparse_values, '')
      elif feature_value.dtype != tf.string:
        feature_value = tf.as_string(feature_value)
      feature_value = tf.reduce_join(feature_value, axis=-1, separator=',')
      outputs[out_name] = feature_value
    return outputs

  def build_rtp_output_dict(self):
    """For exporting: get output nodes for RTP infering."""
    return {}

  def restore(self,
              ckpt_path,
              include_global_step=False,
              ckpt_var_map_path='',
              force_restore_shape_compatible=False):
    """Restore variables from ckpt_path.

    steps:
      1. list the variables in graph that need to be restored
      2. inspect checkpoint and find the variables that could restore from checkpoint
         substitute scope names in case necessary
      3. call tf.train.init_from_checkpoint to restore the variables

    Args:
       ckpt_path: checkpoint path to restore from
       include_global_step: whether to restore global_step variable
       ckpt_var_map_path: variable map from graph variables to variables in a checkpoint
          each line consists of: variable name in graph  variable name in ckpt
       force_restore_shape_compatible: if variable shape is incompatible, clip or pad
          variables in checkpoint, and then restore

    Returns:
      IncompatibleShapeRestoreHook if force_shape_compatible else None
    """
    name2var_map = self._get_restore_vars(ckpt_var_map_path)
    logging.info('start to restore from %s' % ckpt_path)

    ckpt_reader = tf.train.NewCheckpointReader(ckpt_path)
    ckpt_var2shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
      ckpt_var2shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)

    vars_in_ckpt = {}
    incompatible_shape_var_map = {}
    fail_restore_vars = []
    for variable_name, variable in sorted(name2var_map.items()):
      if variable_name in ckpt_var2shape_map:
        print('restore %s' % variable_name)
        ckpt_var_shape = ckpt_var2shape_map[variable_name]
        if type(variable) == list:
          shape_arr = [x.get_shape() for x in variable]
          var_shape = list(shape_arr[0])
          for x in shape_arr[1:]:
            var_shape[0] += x[0]
          var_shape = tensor_shape.TensorShape(var_shape)
          variable = variables.PartitionedVariable(
              variable_name,
              var_shape,
              variable[0].dtype,
              variable,
              partitions=[len(variable)] + [1] * (len(var_shape) - 1))
        else:
          var_shape = variable.shape.as_list()
        if ckpt_var_shape == var_shape:
          vars_in_ckpt[variable_name] = list(variable) if isinstance(
              variable, variables.PartitionedVariable) else variable
        elif len(ckpt_var_shape) == len(var_shape):
          if force_restore_shape_compatible:
            # create a variable compatible with checkpoint to restore
            dtype = variable[0].dtype if isinstance(variable,
                                                    list) else variable.dtype
            with tf.variable_scope('incompatible_shape_restore'):
              tmp_var = tf.get_variable(
                  name=variable_name + '_T_E_M_P',
                  shape=ckpt_var_shape,
                  trainable=False,
                  # add to a special collection for easy reference
                  # by tf.get_collection('T_E_M_P_RESTROE')
                  collections=['T_E_M_P_RESTROE'],
                  dtype=dtype)
            vars_in_ckpt[variable_name] = tmp_var
            incompatible_shape_var_map[variable] = tmp_var
            print('incompatible restore %s[%s, %s]' %
                  (variable_name, str(var_shape), str(ckpt_var_shape)))
          else:
            logging.warning(
                'Variable [%s] is available in checkpoint, but '
                'incompatible shape with model variable.', variable_name)
        else:
          logging.warning(
              'Variable [%s] is available in checkpoint, but '
              'incompatible shape dims with model variable.', variable_name)
      elif 'EmbeddingVariable' in str(type(variable)):
        if '%s-keys' % variable_name not in ckpt_var2shape_map:
          continue
        print('restore embedding_variable %s' % variable_name)
        from tensorflow.python.training import saver
        names_to_saveables = saver.BaseSaverBuilder.OpListToDict([variable])
        saveable_objects = []
        for name, op in names_to_saveables.items():
          for s in saver.BaseSaverBuilder.SaveableObjectsForOp(op, name):
            saveable_objects.append(s)
        init_op = saveable_objects[0].restore([ckpt_path], None)
        variable._initializer_op = init_op
      elif type(variable) == list and 'EmbeddingVariable' in str(
          type(variable[0])):
        if '%s/part_0-keys' % variable_name not in ckpt_var2shape_map:
          continue
        print('restore partitioned embedding_variable %s' % variable_name)
        from tensorflow.python.training import saver
        for part_var in variable:
          names_to_saveables = saver.BaseSaverBuilder.OpListToDict([part_var])
          saveable_objects = []
          for name, op in names_to_saveables.items():
            for s in saver.BaseSaverBuilder.SaveableObjectsForOp(op, name):
              saveable_objects.append(s)
          init_op = saveable_objects[0].restore([ckpt_path], None)
          part_var._initializer_op = init_op
      elif sok is not None and isinstance(variable, sok.DynamicVariable):
        print('restore dynamic_variable %s' % variable_name)
        keys, vals = load_embed_lib.load_kv_embed(
            task_index=hvd.rank(),
            task_num=hvd.size(),
            embed_dim=variable._dimension,
            var_name='embed-' + variable.name.replace('/', '__'),
            ckpt_path=ckpt_path)
        with ops.control_dependencies([variable._initializer_op]):
          variable._initializer_op = dynamic_variable_ops.dummy_var_assign(
              variable.handle, keys, vals)
      else:
        fail_restore_vars.append(variable_name)
    for variable_name in fail_restore_vars:
      if 'Momentum' not in variable_name:
        logging.warning('Variable [%s] is not available in checkpoint',
                        variable_name)

    tf.train.init_from_checkpoint(ckpt_path, vars_in_ckpt)

    if force_restore_shape_compatible:
      return estimator_utils.IncompatibleShapeRestoreHook(
          incompatible_shape_var_map)
    else:
      return None

  def _get_restore_vars(self, ckpt_var_map_path):
    """Restore by specify variable map between graph variables and ckpt variables.

    Args:
      ckpt_var_map_path: variable map from graph variables to variables in a checkpoint
          each line consists of: variable name in graph  variable name in ckpt

    Returns:
      the list of variables which need to restore from checkpoint
    """
    # here must use global_variables, because variables such as moving_mean
    #  and moving_variance is usually not trainable in detection models
    all_vars = tf.global_variables()
    PARTITION_PATTERN = '/part_[0-9]+'
    VAR_SUFIX_PATTERN = ':[0-9]$'

    name2var = {}
    for one_var in all_vars:
      var_name = re.sub(VAR_SUFIX_PATTERN, '', one_var.name)
      if re.search(PARTITION_PATTERN,
                   var_name) and one_var._save_slice_info is not None:
        var_name = re.sub(PARTITION_PATTERN, '', var_name)
        is_part = True
      else:
        is_part = False
      if var_name in name2var:
        assert is_part, 'multiple vars: %s' % var_name
        name2var[var_name].append(one_var)
      else:
        name2var[var_name] = [one_var] if is_part else one_var

    if ckpt_var_map_path != '':
      if not gfile.Exists(ckpt_var_map_path):
        logging.warning('%s not exist' % ckpt_var_map_path)
        return name2var

      # load var map
      name_map = {}
      with gfile.GFile(ckpt_var_map_path, 'r') as fin:
        for one_line in fin:
          one_line = one_line.strip()
          line_tok = [x for x in one_line.split() if x != '']
          if len(line_tok) != 2:
            logging.warning('Failed to process: %s' % one_line)
            continue
          name_map[line_tok[0]] = line_tok[1]
      update_map = {}
      old_keys = []
      for var_name in name2var:
        if var_name in name_map:
          in_ckpt_name = name_map[var_name]
          update_map[in_ckpt_name] = name2var[var_name]
          old_keys.append(var_name)
      for tmp_key in old_keys:
        del name2var[tmp_key]
      name2var.update(update_map)
      return name2var
    else:
      var_filter, scope_update = self.get_restore_filter()
      if var_filter is not None:
        name2var = {
            var_name: name2var[var_name]
            for var in name2var
            if var_filter.keep(var.name)
        }
      # drop scope prefix if necessary
      if scope_update is not None:
        name2var = {
            scope_update(var_name): name2var[var_name] for var_name in name2var
        }
      return name2var

  def get_restore_filter(self):
    """Get restore variable filter.

    Return:
       filter: type of Filter in restore_filter.py
       scope_drop: type of ScopeDrop in restore_filter.py
    """
    if len(self._base_model_config.restore_filters) == 0:
      return None, None

    for x in self._base_model_config.restore_filters:
      logging.info('restore will filter out pattern %s' % x)

    all_filters = [
        restore_filter.KeywordFilter(x, True)
        for x in self._base_model_config.restore_filters
    ]

    return restore_filter.CombineFilter(all_filters,
                                        restore_filter.Logical.AND), None

  def get_grouped_vars(self, opt_num):
    """Group the vars into different optimization groups.

    Each group will be optimized by a separate optimizer.

    Args:
      opt_num: number of optimizers from easyrec config.

    Return:
      list of list of variables.
    """
    assert opt_num == 2, 'could only support 2 optimizers, one for embedding, one for the other layers'

    embedding_vars = []
    deep_vars = []
    for tmp_var in variables.trainable_variables():
      if tmp_var.name.startswith(
          'input_layer') or '/embedding_weights' in tmp_var.name:
        embedding_vars.append(tmp_var)
      else:
        deep_vars.append(tmp_var)
    return [embedding_vars, deep_vars]
