# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

import tensorflow as tf

from easy_rec.python.input.odps_rtp_input import OdpsRTPInput

try:
  import pai
  import rtp_fg
except Exception:
  pai = None
  rtp_fg = None


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
               fg_json_path=None,
               pipeline_config=None):
    super(OdpsRTPInputV2,
          self).__init__(data_config, feature_config, input_path, task_index,
                         task_num, check_mode, pipeline_config)
    if fg_json_path.startswith('!'):
      fg_json_path = fg_json_path[1:]
    self._fg_config_path = fg_json_path
    logging.info('fg config path: {}'.format(self._fg_config_path))
    if self._fg_config_path is None:
      raise ValueError('fg_json_path is not set')
    with tf.gfile.GFile(self._fg_config_path, 'r') as f:
      self._fg_config = json.load(f)

  def _parse_table(self, *fields):
    self.check_rtp()

    fields = list(fields)
    labels = fields[:-1]

    # assume that the last field is the generated feature column
    features = rtp_fg.parse_genreated_fg(self._fg_config, fields[-1])

    field_keys = [x for x in self._input_fields if x not in self._label_fields]
    for feature_key in features:
      if feature_key not in field_keys or feature_key not in self._effective_fields:
        del features[feature_key]
    inputs = {x: features[x] for x in features.keys()}

    for x in range(len(self._label_fields)):
      inputs[self._label_fields[x]] = labels[x]
    return inputs

  def create_placeholders(self, *args, **kwargs):
    """Create serving placeholders with rtp_fg."""
    self.check_rtp()
    self._mode = tf.estimator.ModeKeys.PREDICT
    inputs_placeholder = tf.placeholder(tf.string, [None], name='features')
    print('[OdpsRTPInputV2] building placeholders.')
    print('[OdpsRTPInputV2] fg_config: {}'.format(self._fg_config))
    features = rtp_fg.parse_genreated_fg(self._fg_config, inputs_placeholder)
    print('[OdpsRTPInputV2] built features: {}'.format(features.keys()))
    features = self._preprocess(features)
    print('[OdpsRTPInputV2] processed features: {}'.format(features.keys()))
    return {'features': inputs_placeholder}, features['feature']

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
