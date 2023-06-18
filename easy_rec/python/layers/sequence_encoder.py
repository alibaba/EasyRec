# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import tensorflow as tf

from easy_rec.python.layers.keras.bst import BST
from easy_rec.python.layers.keras.din import DIN
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class SequenceEncoder(object):

  def __init__(self, input_layer, feature_configs, feature_groups_config,
               l2_reg):
    self._input_layer = input_layer
    self._feature_groups_config = {
        x.group_name: x for x in feature_groups_config
    }
    self._l2_reg = l2_reg
    self._feature_config_by_name = {
        x.feature_name if x.HasField('feature_name') else x.input_names[0]: x
        for x in feature_configs
    }

    for name, group in self._feature_groups_config.items():
      if len(group.sequence_encoders) == 0:
        continue
      check_share_emb = False
      for encoder in group.sequence_encoders:
        if encoder.force_share_embeddings:
          check_share_emb = True
          break
      if not check_share_emb:
        continue
      if not self.check_share_embedding(group):
        raise ValueError(
            'sequence feature group `%s` check share embedding failed, '
            'you should add `embedding_name` to feature config' % name)

  def check_share_embedding(self, feature_group):
    seq_emb_names = set()
    target_emb_names = set()
    for feature in feature_group.feature_names:
      conf = self._feature_config_by_name[feature]
      if not conf.HasField('embedding_name'):
        return False
      if conf.feature_type == FeatureConfig.FeatureType.SequenceFeature:
        seq_emb_names.add(conf.embedding_name)
      else:
        target_emb_names.add(conf.embedding_name)

    if seq_emb_names != target_emb_names:
      tf.logging.error(
          'sequence share embedding names: %s, target share embedding names: %s'
          % (','.join(seq_emb_names), ','.join(target_emb_names)))
      return False
    return True

  def __call__(self, features, group_name, is_training=True, *args, **kwargs):
    group_config = self._feature_groups_config[group_name]
    if len(group_config.sequence_encoders) == 0:
      return None

    seq_features, target_feature, target_features = self._input_layer(
        features, group_name, is_combine=False)
    assert len(
        seq_features) > 0, 'sequence feature is empty in group: ' + group_name

    outputs = []
    for encoder in group_config.sequence_encoders:
      encoder_type = encoder.WhichOneof('encoder').lower()
      if encoder_type == 'bst':
        bst = BST(encoder.bst, self._l2_reg, name=group_name)
        encoding = bst([seq_features, target_feature], is_training, **kwargs)
        outputs.append(encoding)
      elif encoder_type == 'din':
        din = DIN(encoder.din, self._l2_reg, name=group_name)
        encoding = din([seq_features, target_feature], is_training)
        outputs.append(encoding)
      else:
        assert False, 'unsupported sequence encode type: ' + encoder_type

    if len(outputs) == 0:
      logging.warning(
          "there's no sequence encoder configured in feature group: " +
          group_name)
      return None
    if len(outputs) == 1:
      return outputs[0]

    return tf.concat(outputs, axis=-1)
