# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
from enum import Enum
from easy_rec.python.utils.input_utils import concat_parsed_features

import tensorflow as tf
from tensorflow.contrib.framework import argsort as tf_argsort

from easy_rec.python.input.odps_rtp_input import OdpsRTPInput

try:
  import pai
  import rtp_fg
except Exception:
  pai = None
  rtp_fg = None


class RtpFeatureType(Enum):
  RAW_FEATURE = "raw_feature"
  ID_FEATURE = "id_feature"
  COMBO_FEATURE = "combo_feature"
  LOOKUP_FEATURE = "lookup_feature"
  MATCH_FEATURE = "match_feature"


class RtpFeatureConfig:
  def __init__(self, fc_dict):
    self.feature_name = str(fc_dict.get('feature_name'))
    self.feature_type = RtpFeatureType(fc_dict.get('feature_type'))
    self.value_dimension = int(fc_dict.get('value_dimension', 0))


class RtpSequenceConfig:
  def __init__(self, fc_dict):
    self.sequence_name = str(fc_dict.get('sequence_name'))
    self.sequence_length = int(fc_dict.get('sequence_length'))
    if self.sequence_length <= 0:
      raise ValueError("sequence feature [{}] has illegal sequence length [{}]"\
        .format(self.sequence_name, self.sequence_length))
    self.features = [RtpFeatureConfig(feature_dict) for feature_dict in fc_dict.get('features')]


def parse_rtp_feature_config(fg_config_dict):
  feature_configs = []
  for fc_dict in fg_config_dict.get('features'):
    if fc_dict.get('sequence_name'):
      feature_configs.append(RtpSequenceConfig(fc_dict))
    else:
      feature_configs.append(RtpFeatureConfig(fc_dict))
  return feature_configs


class OdpsRTPInputV2(OdpsRTPInput):
  """RTPInput for parsing rtp fg new input format on odps.

  Our new format(csv in table) of rtp output:
     label0, item_id, ..., user_id, features
  Where features is in default RTP-tensorflow format.
  The features column and labels are specified by data_config.selected_cols,
     columns are selected by names in the table
     such as: clk,features, the last selected column is features, the first
     selected columns are labels
  """

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False,
               fg_json_path=None):
    super(OdpsRTPInputV2,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode)
    if fg_json_path.startswith('!'):
      fg_json_path = fg_json_path[1:]
    self._fg_config_path = fg_json_path
    logging.info('fg config path: {}'.format(self._fg_config_path))
    if self._fg_config_path is None:
      raise ValueError('fg_json_path is not set')
    with tf.gfile.GFile(self._fg_config_path, 'r') as f:
      self._fg_config = json.load(f)
    self._rtp_features = parse_rtp_feature_config(self._fg_config)

  def _preprocess(self, field_dict):
    parsed_dict = {}
    neg_samples = self._maybe_negative_sample(field_dict)
    neg_parsed_dict = {}
    if neg_samples:
      neg_field_dict = {}
      for k, v in neg_samples.items():
        if k in field_dict:
          neg_field_dict[k] = v
        else:
          print('appended fields: %s' % k)
          parsed_dict[k] = v
          self._appended_fields.append(k)
      neg_parsed_dict = self._preprocess_without_negative_sample(neg_field_dict,
        ignore_absent_fields=True)
    for k, v in self._preprocess_without_negative_sample(field_dict).items():
      if k in neg_parsed_dict:
        try:
          v = concat_parsed_features([v, neg_parsed_dict[k]], name=k)
        except Exception as e:
          logging.error("failed to concat parsed features [{}]".format(k))
          raise
      parsed_dict[k] = v
    return parsed_dict

  def _parse_table(self, *fields):
    self.check_rtp()

    fields = list(fields)
    labels = fields[:-1]

    # assume that the last field is the generated feature column
    features = rtp_fg.parse_genreated_fg(self._fg_config, fields[-1])

    inputs = self._transform_features(features)

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]

    return inputs

  def _transform_features(self, rtp_features):
    """Transform features from RTP format into EasyRec format."""
    features = {}
    for fc in self._rtp_features:
      if isinstance(fc, RtpSequenceConfig):
        for sfc in fc.features:
          sub_feature_name = "{}__{}".format(fc.sequence_name, sfc.feature_name)
          with tf.name_scope('sequence_feature_transform/{}'.format(sub_feature_name)):
            shape_0_list = []
            shape_2_list = []
            indices_0_list = []
            indices_1_list = []
            indices_2_list = []
            values_list = []
            if sfc.feature_type == RtpFeatureType.ID_FEATURE:
              for i in range(fc.sequence_length):
                sub_feature_name_rtp = "{}_{}_{}".format(fc.sequence_name, i, sfc.feature_name)
                if sub_feature_name_rtp not in rtp_features:
                  raise ValueError("sequence sub feature [{}] is missing"\
                    .format(sub_feature_name_rtp))
                sub_feature_tensor = rtp_features[sub_feature_name_rtp]
                assert isinstance(sub_feature_tensor, tf.SparseTensor), \
                  "sequence sub feature [{}] must be sparse"
                values_list.append(sub_feature_tensor.values)
                shape_0_list.append(sub_feature_tensor.dense_shape[0])
                shape_2_list.append(sub_feature_tensor.dense_shape[1])
                indices_0_item = sub_feature_tensor.indices[:,0]
                indices_1_item = tf.tile(tf.constant([i], dtype=indices_0_item.dtype),
                                         tf.shape(indices_0_item))
                indices_2_item = sub_feature_tensor.indices[:,1]
                indices_0_list.append(indices_0_item)
                indices_1_list.append(indices_1_item)
                indices_2_list.append(indices_2_item)
            elif sfc.feature_type == RtpFeatureType.RAW_FEATURE:
              for i in range(fc.sequence_length):
                sub_feature_name_rtp = "{}_{}_{}".format(fc.sequence_name, i, sfc)
                if sub_feature_name_rtp not in rtp_features:
                  raise ValueError("sequence sub feature [{}] is missing"\
                    .format(sub_feature_name_rtp))
                sub_feature_tensor = rtp_features[sub_feature_name_rtp]
                assert isinstance(sub_feature_tensor, tf.Tensor), \
                  "sequence sub feature [{}] must be dense".format(sub_feature_name_rtp)
                values_list.append(sub_feature_tensor)
                assert len(sub_feature_tensor.get_shape()) == 2, \
                  "sequence sub feature [{}] must be 2-dimensional".format(sub_feature_name_rtp)
                sub_feature_shape = tf.shape(sub_feature_tensor)
                sub_feature_shape_0 = sub_feature_shape[0]
                sub_feature_shape_1 = sub_feature_shape[1]
                shape_0_list.append(sub_feature_shape_0)
                shape_2_list.append(sub_feature_shape_1)
                indices_2_item, indices_0_item = tf.meshgrid(
                  tf.range(0, sub_feature_shape_1),
                  tf.range(0, sub_feature_shape_0))
                num_elements = tf.reduce_prod(sub_feature_shape)
                indices_0_item = tf.reshape(indices_0_item, [num_elements])
                indices_1_item = tf.tile(tf.constant([i], dtype=indices_0_item.dtype),
                                         tf.constant([num_elements], dtype=tf.int32))
                indices_2_item = tf.reshape(indices_2_item, [num_elements])
                indices_0_list.append(indices_0_item)
                indices_1_list.append(indices_1_item)
                indices_2_list.append(indices_2_item)
            else:
              raise ValueError("sequence sub feature [{}] illegal type [{}]"\
                .format(sub_feature_name, sfc.feature_type))
            # note that, as the first dimension is batch size, all values in shape_0_list should be the same
            indices_0 = tf.concat(indices_0_list, axis=0, name='indices_0')
            shape_0 = tf.reduce_max(shape_0_list, name='shape_0')
            # the second dimension is the sequence length
            indices_1 = tf.concat(indices_1_list, axis=0, name='indices_1')
            shape_1 = tf.maximum(tf.add(tf.reduce_max(indices_1), 1), 0, name='shape_1')
            # shape_2 is the max number of multi-values of a single feature value
            indices_2 = tf.concat(indices_2_list, axis=0, name='indices_2')
            shape_2 = tf.reduce_max(shape_2_list, name='shape_2')
            # values
            values = tf.concat(values_list, axis=0, name='values')
            # sort the values along the first dimension indices
            sorting = tf_argsort(indices_0, name='argsort_after_concat')
            is_single_sample = tf.equal(shape_0, tf.constant(1, dtype=shape_0.dtype), name='is_single_sample')
            indices_0 = tf.cond(is_single_sample,
              lambda: indices_0,
              lambda: tf.gather(indices_0, sorting, name='indices_0_sorted'),
              name='indices_0_optional')
            indices_1 = tf.cond(is_single_sample,
              lambda: indices_1,
              lambda: tf.gather(indices_1, sorting, name='indices_1_sorted'),
              name='indices_1_optional')
            indices_2 = tf.cond(is_single_sample,
              lambda: indices_2,
              lambda: tf.gather(indices_2, sorting, name='indices_2_sorted'),
              name='indices_2_optional')
            values = tf.cond(is_single_sample,
              lambda: values,
              lambda: tf.gather(values, sorting, name='values_sorted'),
              name='values_optional')
            # construct the 3-dimensional sparse tensor
            features[sub_feature_name] = tf.SparseTensor(
              dense_shape=tf.stack([shape_0, shape_1, shape_2], axis=0, name='shape'),
              indices=tf.stack([indices_0, indices_1, indices_2], axis=1, name='indices'),
              values=values
            )
      elif isinstance(fc, RtpFeatureConfig):
        features[fc.feature_name] = rtp_features[fc.feature_name]
      else:
        raise TypeError("illegal feature config type {}".format(type(fc)))
    return features

  def create_placeholders(self, *args, **kwargs):
    """Create serving placeholders with rtp_fg."""
    self.check_rtp()
    self._mode = tf.estimator.ModeKeys.PREDICT
    inputs_placeholder = tf.placeholder(tf.string, [None], name='features')
    print('[OdpsRTPInputV2] building placeholders.')
    print('[OdpsRTPInputV2] fg_config: {}'.format(self._fg_config))
    features = rtp_fg.parse_genreated_fg(self._fg_config, inputs_placeholder)
    features = self._transform_features(features)
    print('[OdpsRTPInputV2] built features: {}'.format(features.keys()))
    features = self._preprocess(features)
    print('[OdpsRTPInputV2] processed features: {}'.format(features.keys()))
    return {'features': inputs_placeholder}, features

  def create_multi_placeholders(self, *args, **kwargs):
    """Create serving multi-placeholders with rtp_fg."""
    raise NotImplementedError(
        'create_multi_placeholders is not supported for OdpsRTPInputV2')

  def check_rtp(self):
    if rtp_fg is None:
      raise NotImplementedError(
          'OdpsRTPInputV2 cannot run without rtp_fg, which is not installed')

  def _pre_build(self, mode, params):
    try:
      # Prevent TF from replacing the shape tensor to a constant tensor. This will
      # cause the batch size being fixed. And RTP will be not able to recognize
      # the input shape.
      tf.get_default_graph().set_shape_optimize(False)
    except AttributeError as e:
      logging.warning('failed to disable shape optimization:', e)
    
