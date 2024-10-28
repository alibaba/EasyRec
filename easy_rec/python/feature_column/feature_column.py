# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import logging
import sys

import tensorflow as tf
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.platform import gfile

from easy_rec.python.builders import hyperparams_builder
from easy_rec.python.compat.feature_column import sequence_feature_column
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep
from easy_rec.python.utils.proto_util import copy_obj

from easy_rec.python.compat.feature_column import feature_column_v2 as feature_column  # NOQA

MAX_HASH_BUCKET_SIZE = 9223372036854775807


class FeatureKeyError(KeyError):

  def __init__(self, feature_name):
    super(FeatureKeyError, self).__init__(feature_name)


class SharedEmbedding(object):

  def __init__(self, embedding_name, index, sequence_combiner=None):
    self.embedding_name = embedding_name
    self.index = index
    self.sequence_combiner = sequence_combiner


EVParams = collections.namedtuple('EVParams', [
    'filter_freq', 'steps_to_live', 'use_cache', 'init_capacity', 'max_capacity'
])


class FeatureColumnParser(object):
  """Parse and generate feature columns."""

  def __init__(self,
               feature_configs,
               wide_deep_dict={},
               wide_output_dim=-1,
               ev_params=None):
    """Initializes a `FeatureColumnParser`.

    Args:
      feature_configs: collections of
            easy_rec.python.protos.feature_config_pb2.FeatureConfig
            or easy_rec.python.protos.feature_config_pb2.FeatureConfigV2.features
      wide_deep_dict: dict of {feature_name:WideOrDeep}, passed by
        easy_rec.python.layers.input_layer.InputLayer, it is defined in
        easy_rec.python.protos.easy_rec_model_pb2.EasyRecModel.feature_groups
      wide_output_dim: output dimension for wide columns
      ev_params: params used by EmbeddingVariable, which is provided by pai-tf
    """
    self._feature_configs = feature_configs
    self._wide_output_dim = wide_output_dim
    self._wide_deep_dict = wide_deep_dict
    self._deep_columns = {}
    self._wide_columns = {}
    self._sequence_columns = {}

    self._share_embed_names = {}
    self._share_embed_infos = {}

    self._vocab_size = {}

    self._global_ev_params = None
    if ev_params is not None:
      self._global_ev_params = self._build_ev_params(ev_params)

    def _cmp_embed_config(a, b):
      return a.embedding_dim == b.embedding_dim and a.combiner == b.combiner and\
          a.initializer == b.initializer and a.max_partitions == b.max_partitions and\
          a.embedding_name == b.embedding_name

    for config in self._feature_configs:
      if not config.HasField('embedding_name'):
        continue
      embed_name = config.embedding_name

      if embed_name in self._share_embed_names:
        assert _cmp_embed_config(config, self._share_embed_infos[embed_name]),\
            'shared embed info of [%s] is not matched [%s] vs [%s]' % (
                embed_name, config, self._share_embed_infos[embed_name])
        self._share_embed_names[embed_name] += 1
        if config.feature_type == FeatureConfig.FeatureType.SequenceFeature:
          self._share_embed_infos[embed_name] = copy_obj(config)
      else:
        self._share_embed_names[embed_name] = 1
        self._share_embed_infos[embed_name] = copy_obj(config)

    # remove not shared embedding names
    not_shared = [
        x for x in self._share_embed_names if self._share_embed_names[x] == 1
    ]
    for embed_name in not_shared:
      del self._share_embed_names[embed_name]
      del self._share_embed_infos[embed_name]

    logging.info('shared embeddings[num=%d]' % len(self._share_embed_names))
    for embed_name in self._share_embed_names:
      logging.info('\t%s: share_num[%d], share_info[%s]' %
                   (embed_name, self._share_embed_names[embed_name],
                    self._share_embed_infos[embed_name]))
    self._deep_share_embed_columns = {
        embed_name: [] for embed_name in self._share_embed_names
    }
    self._wide_share_embed_columns = {
        embed_name: [] for embed_name in self._share_embed_names
    }

    self._feature_vocab_size = {}
    for config in self._feature_configs:
      assert isinstance(config, FeatureConfig)
      try:
        if config.feature_type == config.IdFeature:
          self.parse_id_feature(config)
        elif config.feature_type == config.TagFeature:
          self.parse_tag_feature(config)
        elif config.feature_type == config.RawFeature:
          self.parse_raw_feature(config)
        elif config.feature_type == config.ComboFeature:
          self.parse_combo_feature(config)
        elif config.feature_type == config.LookupFeature:
          self.parse_lookup_feature(config)
        elif config.feature_type == config.SequenceFeature:
          self.parse_sequence_feature(config)
        elif config.feature_type == config.ExprFeature:
          self.parse_expr_feature(config)
        elif config.feature_type != config.PassThroughFeature:
          assert False, 'invalid feature type: %s' % config.feature_type
      except FeatureKeyError:
        pass

    for embed_name in self._share_embed_names:
      initializer = None
      if self._share_embed_infos[embed_name].HasField('initializer'):
        initializer = hyperparams_builder.build_initializer(
            self._share_embed_infos[embed_name].initializer)

      partitioner = self._build_partitioner(self._share_embed_infos[embed_name])

      if self._share_embed_infos[embed_name].HasField('ev_params'):
        ev_params = self._build_ev_params(
            self._share_embed_infos[embed_name].ev_params)
      else:
        ev_params = self._global_ev_params

      # for handling share embedding columns
      if len(self._deep_share_embed_columns[embed_name]) > 0:
        share_embed_fcs = feature_column.shared_embedding_columns(
            self._deep_share_embed_columns[embed_name],
            self._share_embed_infos[embed_name].embedding_dim,
            initializer=initializer,
            shared_embedding_collection_name=embed_name,
            combiner=self._share_embed_infos[embed_name].combiner,
            partitioner=partitioner,
            ev_params=ev_params)
        config = self._share_embed_infos[embed_name]
        max_seq_len = config.max_seq_len if config.HasField(
            'max_seq_len') else -1
        for fc in share_embed_fcs:
          fc.max_seq_length = max_seq_len
        self._deep_share_embed_columns[embed_name] = share_embed_fcs

      # for handling wide share embedding columns
      if len(self._wide_share_embed_columns[embed_name]) > 0:
        share_embed_fcs = feature_column.shared_embedding_columns(
            self._wide_share_embed_columns[embed_name],
            self._wide_output_dim,
            initializer=initializer,
            shared_embedding_collection_name=embed_name + '_wide',
            combiner='sum',
            partitioner=partitioner,
            ev_params=ev_params)
        config = self._share_embed_infos[embed_name]
        max_seq_len = config.max_seq_len if config.HasField(
            'max_seq_len') else -1
        for fc in share_embed_fcs:
          fc.max_seq_length = max_seq_len
        self._wide_share_embed_columns[embed_name] = share_embed_fcs

    for fc_name in self._deep_columns:
      fc = self._deep_columns[fc_name]
      if isinstance(fc, SharedEmbedding):
        self._deep_columns[fc_name] = self._get_shared_embedding_column(fc)

    for fc_name in self._wide_columns:
      fc = self._wide_columns[fc_name]
      if isinstance(fc, SharedEmbedding):
        self._wide_columns[fc_name] = self._get_shared_embedding_column(
            fc, deep=False)

    for fc_name in self._sequence_columns:
      fc = self._sequence_columns[fc_name]
      if isinstance(fc, SharedEmbedding):
        self._sequence_columns[fc_name] = self._get_shared_embedding_column(fc)

  @property
  def wide_columns(self):
    return self._wide_columns

  @property
  def deep_columns(self):
    return self._deep_columns

  @property
  def sequence_columns(self):
    return self._sequence_columns

  def is_wide(self, config):
    if config.HasField('feature_name'):
      feature_name = config.feature_name
    else:
      feature_name = config.input_names[0]
    if feature_name not in self._wide_deep_dict:
      raise FeatureKeyError(feature_name)
    return self._wide_deep_dict[feature_name] in [
        WideOrDeep.WIDE, WideOrDeep.WIDE_AND_DEEP
    ]

  def is_deep(self, config):
    if config.HasField('feature_name'):
      feature_name = config.feature_name
    else:
      feature_name = config.input_names[0]
    # DEEP or WIDE_AND_DEEP
    if feature_name not in self._wide_deep_dict:
      raise FeatureKeyError(feature_name)
    return self._wide_deep_dict[feature_name] in [
        WideOrDeep.DEEP, WideOrDeep.WIDE_AND_DEEP
    ]

  def get_feature_vocab_size(self, feature):
    return self._feature_vocab_size.get(feature, 1)

  def _get_vocab_size(self, vocab_path):
    if vocab_path in self._vocab_size:
      return self._vocab_size[vocab_path]
    with gfile.GFile(vocab_path, 'r') as fin:
      vocabulary_size = sum(1 for _ in fin)
      self._vocab_size[vocab_path] = vocabulary_size
      return vocabulary_size

  def _get_hash_bucket_size(self, config):
    if not config.HasField('hash_bucket_size'):
      return -1
    if self._global_ev_params is not None or config.HasField('ev_params'):
      return MAX_HASH_BUCKET_SIZE
    else:
      return config.hash_bucket_size

  def parse_id_feature(self, config):
    """Generate id feature columns.

    if hash_bucket_size or vocab_list or vocab_file is set,
    then will accept input tensor of string type, otherwise will accept input
    tensor of integer type.

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    hash_bucket_size = self._get_hash_bucket_size(config)
    if hash_bucket_size > 0:
      fc = feature_column.categorical_column_with_hash_bucket(
          feature_name,
          hash_bucket_size=hash_bucket_size,
          feature_name=feature_name)
    elif config.vocab_list:
      fc = feature_column.categorical_column_with_vocabulary_list(
          feature_name,
          default_value=0,
          vocabulary_list=config.vocab_list,
          feature_name=feature_name)
    elif config.vocab_file:
      fc = feature_column.categorical_column_with_vocabulary_file(
          feature_name,
          default_value=0,
          vocabulary_file=config.vocab_file,
          vocabulary_size=self._get_vocab_size(config.vocab_file),
          feature_name=feature_name)
    else:
      use_ev = self._global_ev_params or config.HasField('ev_params')
      num_buckets = sys.maxsize if use_ev else config.num_buckets
      fc = feature_column.categorical_column_with_identity(
          feature_name, num_buckets, default_value=0, feature_name=feature_name)

    if self.is_wide(config):
      self._add_wide_embedding_column(fc, config)
    if self.is_deep(config):
      self._add_deep_embedding_column(fc, config)

  def parse_tag_feature(self, config):
    """Generate tag feature columns.

    if hash_bucket_size is set, will accept input of SparseTensor of string,
    otherwise num_buckets must be set, will accept input of SparseTensor of integer.
    tag feature preprocess is done in easy_rec/python/input/input.py: Input. _preprocess

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    hash_bucket_size = self._get_hash_bucket_size(config)
    if hash_bucket_size > 0:
      tag_fc = feature_column.categorical_column_with_hash_bucket(
          feature_name,
          hash_bucket_size,
          dtype=tf.string,
          feature_name=feature_name)
    elif config.vocab_list:
      tag_fc = feature_column.categorical_column_with_vocabulary_list(
          feature_name,
          default_value=0,
          vocabulary_list=config.vocab_list,
          feature_name=feature_name)
    elif config.vocab_file:
      tag_fc = feature_column.categorical_column_with_vocabulary_file(
          feature_name,
          default_value=0,
          vocabulary_file=config.vocab_file,
          vocabulary_size=self._get_vocab_size(config.vocab_file),
          feature_name=feature_name)
    else:
      use_ev = self._global_ev_params or config.HasField('ev_params')
      num_buckets = sys.maxsize if use_ev else config.num_buckets
      tag_fc = feature_column.categorical_column_with_identity(
          feature_name, num_buckets, default_value=0, feature_name=feature_name)

    if len(config.input_names) > 1:
      tag_fc = feature_column.weighted_categorical_column(
          tag_fc, weight_feature_key=feature_name + '_w', dtype=tf.float32)
    elif config.HasField('kv_separator'):
      tag_fc = feature_column.weighted_categorical_column(
          tag_fc, weight_feature_key=feature_name + '_w', dtype=tf.float32)

    if self.is_wide(config):
      self._add_wide_embedding_column(tag_fc, config)
    if self.is_deep(config):
      self._add_deep_embedding_column(tag_fc, config)

  def parse_raw_feature(self, config):
    """Generate raw features columns.

    if boundaries is set, will be converted to category_column first.

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    fc = feature_column.numeric_column(
        key=feature_name,
        shape=(config.raw_input_dim,),
        feature_name=feature_name)

    bounds = None
    if config.boundaries:
      bounds = list(config.boundaries)
      bounds.sort()
    elif config.num_buckets > 1 and config.max_val > config.min_val:
      # the feature values are already normalized into [0, 1]
      bounds = [
          x / float(config.num_buckets) for x in range(0, config.num_buckets)
      ]
      logging.info('discrete %s into %d buckets' %
                   (feature_name, config.num_buckets))

    if bounds:
      try:
        fc = feature_column.bucketized_column(fc, bounds)
      except Exception as e:
        logging.error('bucketized_column [%s] with bounds %s error' %
                      (fc.name, str(bounds)))
        raise e
      if self.is_wide(config):
        self._add_wide_embedding_column(fc, config)
      if self.is_deep(config):
        self._add_deep_embedding_column(fc, config)
    else:
      tmp_id_col = feature_column.categorical_column_with_identity(
          feature_name + '_raw_proj_id',
          config.raw_input_dim,
          default_value=0,
          feature_name=feature_name)
      wgt_fc = feature_column.weighted_categorical_column(
          tmp_id_col,
          weight_feature_key=feature_name + '_raw_proj_val',
          dtype=tf.float32)
      if self.is_wide(config):
        self._add_wide_embedding_column(wgt_fc, config)
      if self.is_deep(config):
        if config.embedding_dim > 0:
          self._add_deep_embedding_column(wgt_fc, config)
        else:
          self._deep_columns[feature_name] = fc

  def parse_expr_feature(self, config):
    """Generate raw features columns.

    if boundaries is set, will be converted to category_column first.

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    fc = feature_column.numeric_column(
        feature_name, shape=(1,), feature_name=feature_name)
    if self.is_wide(config):
      self._add_wide_embedding_column(fc, config)
    if self.is_deep(config):
      self._deep_columns[feature_name] = fc

  def parse_combo_feature(self, config):
    """Generate combo feature columns.

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else None
    assert len(config.input_names) >= 2

    if len(config.combo_join_sep) == 0:
      input_names = []
      for input_id in range(len(config.input_names)):
        if input_id == 0:
          input_names.append(feature_name)
        else:
          input_names.append(feature_name + '_' + str(input_id))
      fc = feature_column.crossed_column(
          input_names,
          self._get_hash_bucket_size(config),
          hash_key=None,
          feature_name=feature_name)
    else:
      fc = feature_column.categorical_column_with_hash_bucket(
          feature_name,
          hash_bucket_size=self._get_hash_bucket_size(config),
          feature_name=feature_name)

    if self.is_wide(config):
      self._add_wide_embedding_column(fc, config)
    if self.is_deep(config):
      self._add_deep_embedding_column(fc, config)

  def parse_lookup_feature(self, config):
    """Generate lookup feature columns.

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    assert config.HasField('hash_bucket_size')
    hash_bucket_size = self._get_hash_bucket_size(config)
    fc = feature_column.categorical_column_with_hash_bucket(
        feature_name,
        hash_bucket_size,
        dtype=tf.string,
        feature_name=feature_name)

    if self.is_wide(config):
      self._add_wide_embedding_column(fc, config)
    if self.is_deep(config):
      self._add_deep_embedding_column(fc, config)

  def parse_sequence_feature(self, config):
    """Generate sequence feature columns.

    Args:
      config: instance of easy_rec.python.protos.feature_config_pb2.FeatureConfig
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    sub_feature_type = config.sub_feature_type
    assert sub_feature_type in [config.IdFeature, config.RawFeature], \
        'Current sub_feature_type only support IdFeature and RawFeature.'
    if sub_feature_type == config.IdFeature:
      if config.HasField('hash_bucket_size'):
        hash_bucket_size = self._get_hash_bucket_size(config)
        fc = sequence_feature_column.sequence_categorical_column_with_hash_bucket(
            feature_name,
            hash_bucket_size,
            dtype=tf.string,
            feature_name=feature_name)
      elif config.vocab_list:
        fc = sequence_feature_column.sequence_categorical_column_with_vocabulary_list(
            feature_name,
            default_value=0,
            vocabulary_list=config.vocab_list,
            feature_name=feature_name)
      elif config.vocab_file:
        fc = sequence_feature_column.sequence_categorical_column_with_vocabulary_file(
            feature_name,
            default_value=0,
            vocabulary_file=config.vocab_file,
            vocabulary_size=self._get_vocab_size(config.vocab_file),
            feature_name=feature_name)
      else:
        use_ev = self._global_ev_params or config.HasField('ev_params')
        num_buckets = sys.maxsize if use_ev else config.num_buckets
        fc = sequence_feature_column.sequence_categorical_column_with_identity(
            feature_name,
            num_buckets,
            default_value=0,
            feature_name=feature_name)
    else:  # raw feature
      bounds = None
      fc = sequence_feature_column.sequence_numeric_column(
          feature_name, shape=(1,), feature_name=feature_name)
      if config.hash_bucket_size > 0:
        hash_bucket_size = self._get_hash_bucket_size(config)
        assert sub_feature_type == config.IdFeature, \
            'You should set sub_feature_type to IdFeature to use hash_bucket_size.'
      elif config.boundaries:
        bounds = list(config.boundaries)
        bounds.sort()
      elif config.num_buckets > 1 and config.max_val > config.min_val:
        # the feature values are already normalized into [0, 1]
        bounds = [
            x / float(config.num_buckets) for x in range(0, config.num_buckets)
        ]
        logging.info('sequence feature discrete %s into %d buckets' %
                     (feature_name, config.num_buckets))
      if bounds:
        try:
          fc = sequence_feature_column.sequence_numeric_column_with_bucketized_column(
              fc, bounds)
        except Exception as e:
          logging.error(
              'sequence features bucketized_column [%s] with bounds %s error' %
              (feature_name, str(bounds)))
          raise e
      elif config.hash_bucket_size <= 0:
        if config.embedding_dim > 0:
          tmp_id_col = sequence_feature_column.sequence_categorical_column_with_identity(
              feature_name + '_raw_proj_id',
              config.raw_input_dim,
              default_value=0,
              feature_name=feature_name)
          wgt_fc = sequence_feature_column.sequence_weighted_categorical_column(
              tmp_id_col,
              weight_feature_key=feature_name + '_raw_proj_val',
              dtype=tf.float32)
          fc = wgt_fc
        else:
          fc = sequence_feature_column.sequence_numeric_column_with_raw_column(
              fc, config.sequence_length)

    if config.embedding_dim > 0:
      self._add_deep_embedding_column(fc, config)
    else:
      self._sequence_columns[feature_name] = fc

  def _build_partitioner(self, config):
    if config.max_partitions > 1:
      if self._global_ev_params is not None or config.HasField('ev_params'):
        # pai embedding_variable should use fixed_size_partitioner
        return partitioned_variables.fixed_size_partitioner(
            num_shards=config.max_partitions)
      else:
        return partitioned_variables.min_max_variable_partitioner(
            max_partitions=config.max_partitions)
    else:
      return None

  def _add_shared_embedding_column(self, embedding_name, fc, deep=True):
    if deep:
      curr_id = len(self._deep_share_embed_columns[embedding_name])
      self._deep_share_embed_columns[embedding_name].append(fc)
    else:
      curr_id = len(self._wide_share_embed_columns[embedding_name])
      self._wide_share_embed_columns[embedding_name].append(fc)
    return SharedEmbedding(embedding_name, curr_id, None)

  def _get_shared_embedding_column(self, fc_handle, deep=True):
    embed_name, embed_id = fc_handle.embedding_name, fc_handle.index
    if deep:
      tmp = self._deep_share_embed_columns[embed_name][embed_id]
    else:
      tmp = self._wide_share_embed_columns[embed_name][embed_id]
    tmp.sequence_combiner = fc_handle.sequence_combiner
    return tmp

  def _add_wide_embedding_column(self, fc, config):
    """Generate wide feature columns.

    We use embedding to simulate wide column, which is more efficient than indicator column for
    sparse features
    """
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    assert self._wide_output_dim > 0, 'wide_output_dim is not set'
    if config.embedding_name in self._wide_share_embed_columns:
      wide_fc = self._add_shared_embedding_column(
          config.embedding_name, fc, deep=False)
    else:
      initializer = None
      if config.HasField('initializer'):
        initializer = hyperparams_builder.build_initializer(config.initializer)
      if config.HasField('ev_params'):
        ev_params = self._build_ev_params(config.ev_params)
      else:
        ev_params = self._global_ev_params
      wide_fc = feature_column.embedding_column(
          fc,
          self._wide_output_dim,
          combiner='sum',
          initializer=initializer,
          partitioner=self._build_partitioner(config),
          ev_params=ev_params)
    self._wide_columns[feature_name] = wide_fc

  def _add_deep_embedding_column(self, fc, config):
    """Generate deep feature columns."""
    feature_name = config.feature_name if config.HasField('feature_name') \
        else config.input_names[0]
    assert config.embedding_dim > 0, 'embedding_dim is not set for %s' % feature_name
    self._feature_vocab_size[feature_name] = fc.num_buckets
    if config.embedding_name in self._deep_share_embed_columns:
      fc = self._add_shared_embedding_column(config.embedding_name, fc)
    else:
      initializer = None
      if config.HasField('initializer'):
        initializer = hyperparams_builder.build_initializer(config.initializer)
      if config.HasField('ev_params'):
        ev_params = self._build_ev_params(config.ev_params)
      else:
        ev_params = self._global_ev_params
      fc = feature_column.embedding_column(
          fc,
          config.embedding_dim,
          combiner=config.combiner,
          initializer=initializer,
          partitioner=self._build_partitioner(config),
          ev_params=ev_params)
      fc.max_seq_length = config.max_seq_len if config.HasField(
          'max_seq_len') else -1

    if config.feature_type != config.SequenceFeature:
      self._deep_columns[feature_name] = fc
    else:
      if config.HasField('sequence_combiner'):
        fc.sequence_combiner = config.sequence_combiner
      self._sequence_columns[feature_name] = fc

  def _build_ev_params(self, ev_params):
    """Build embedding_variables params."""
    ev_params = EVParams(
        ev_params.filter_freq,
        ev_params.steps_to_live if ev_params.steps_to_live > 0 else None,
        ev_params.use_cache, ev_params.init_capacity, ev_params.max_capacity)
    return ev_params
