from __future__ import division
from __future__ import print_function

import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.meta_graph import read_meta_graph_file

from easy_rec.python.utils import config_util

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

import matplotlib  # NOQA
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt  # NOQA

tf.app.flags.DEFINE_string('model_type', 'variational_dropout',
                           'feature selection model type')
tf.app.flags.DEFINE_string('config_path', '',
                           'feature selection model config path')
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'feature selection model checkpoint path')
tf.app.flags.DEFINE_string('output_dir', '',
                           'feature selection result directory')
tf.app.flags.DEFINE_integer(
    'topk', 100, 'select topk importance features for each feature group')
tf.app.flags.DEFINE_string('fg_path', '', 'fg config path')
tf.app.flags.DEFINE_bool('visualize', False,
                         'visualization feature selection result or not')
FLAGS = tf.app.flags.FLAGS


class VariationalDropoutFS:

  def __init__(self,
               config_path,
               output_dir,
               topk,
               checkpoint_path=None,
               fg_path=None,
               visualize=False):
    self._config_path = config_path
    self._output_dir = output_dir
    self._topk = topk
    if not tf.gfile.Exists(self._output_dir):
      tf.gfile.MakeDirs(self._output_dir)
    self._checkpoint_path = checkpoint_path
    self._fg_path = fg_path
    self._visualize = visualize

  def process(self):
    tf.logging.info('Loading logit_p of VariationalDropout layer ...')
    feature_dim_dropout_p_map, embedding_wise_variational_dropout = self._feature_dim_dropout_ratio(
    )

    feature_importance_map = {}
    for group_name, feature_dim_dropout_p in feature_dim_dropout_p_map.items():
      tf.logging.info('Calculating %s feature importance ...' % group_name)
      feature_importance = self._get_feature_importance(
          feature_dim_dropout_p, embedding_wise_variational_dropout)
      feature_importance_map[group_name] = feature_importance

      tf.logging.info('Dump %s  feature importance to csv ...' % group_name)
      self._dump_to_csv(feature_importance, group_name)

      if self._visualize:
        tf.logging.info('Visualizing %s feature importance ...' % group_name)
        if embedding_wise_variational_dropout:
          self._visualize_embedding_dim_importance(feature_dim_dropout_p)
        self._visualize_feature_importance(feature_importance, group_name)

    tf.logging.info('Processing model config ...')
    self._process_config(feature_importance_map)

  def _feature_dim_dropout_ratio(self):
    """Get dropout ratio of embedding-wise or feature-wise."""
    config = config_util.get_configs_from_pipeline_file(self._config_path)
    assert config.model_config.HasField(
        'variational_dropout'), 'variational_dropout must be in model_config'

    embedding_wise_variational_dropout = config.model_config.variational_dropout.embedding_wise_variational_dropout

    if self._checkpoint_path is None or len(self._checkpoint_path) == 0:
      checkpoint_path = tf.train.latest_checkpoint(config.model_dir)
    else:
      checkpoint_path = self._checkpoint_path

    meta_graph_def = read_meta_graph_file(checkpoint_path + '.meta')
    features_dimension_map = dict()
    for col_def in meta_graph_def.collection_def[
        'variational_dropout'].bytes_list.value:
      name, features_dimension = json.loads(col_def)
      name = 'all' if name == '' else name
      features_dimension_map[name] = OrderedDict(features_dimension)

    tf.logging.info('Reading checkpoint from %s ...' % checkpoint_path)
    reader = tf.train.NewCheckpointReader(checkpoint_path)

    feature_dim_dropout_p_map = {}
    for feature_group in config.model_config.feature_groups:
      group_name = feature_group.group_name

      logit_p_name = 'logit_p' if group_name == 'all' else 'logit_p_%s' % group_name
      try:
        logit_p = reader.get_tensor(logit_p_name)
      except Exception:
        print('get `logit_p` failed, try to get `backbone/logit_p`')
        logit_p = reader.get_tensor('backbone/' + logit_p_name)
      feature_dims_importance = tf.sigmoid(logit_p)
      with tf.Session() as sess:
        feature_dims_importance = feature_dims_importance.eval(session=sess)

      feature_dim_dropout_p = {}
      if embedding_wise_variational_dropout:
        index_end = 0
        for feature_name, feature_dim in features_dimension_map[
            group_name].items():
          index_start = index_end
          index_end = index_start + feature_dim
          feature_dim_dropout_p[feature_name] = feature_dims_importance[
              index_start:index_end]
      else:
        index = 0
        for feature_name in features_dimension_map[group_name].keys():
          feature_dim_dropout_p[feature_name] = feature_dims_importance[index]
          index += 1

      feature_dim_dropout_p_map[group_name] = feature_dim_dropout_p
    return feature_dim_dropout_p_map, embedding_wise_variational_dropout

  def _get_feature_importance(self, feature_dim_dropout_p,
                              embedding_wise_variational_dropout):
    """Calculate feature importance."""
    if embedding_wise_variational_dropout:
      feature_importance = {}
      for item in feature_dim_dropout_p.items():
        dropout_rate_mean = np.mean(item[1])
        feature_importance[item[0]] = dropout_rate_mean
      feature_importance = OrderedDict(
          sorted(feature_importance.items(), key=lambda e: e[1]))
    else:
      feature_importance = OrderedDict(
          sorted(feature_dim_dropout_p.items(), key=lambda e: e[1]))
    return feature_importance

  def _process_config(self, feature_importance_map):
    """Process model config and fg config with feature selection."""
    excluded_features = set()
    for group_name, feature_importance in feature_importance_map.items():
      for i, (feature_name, _) in enumerate(feature_importance.items()):
        if i >= self._topk:
          excluded_features.add(feature_name)

    config = config_util.get_configs_from_pipeline_file(self._config_path)
    # keep sequence features and side-infos
    sequence_features = set()
    for feature_group in config.model_config.feature_groups:
      for sequence_feature in feature_group.sequence_features:
        for seq_att_map in sequence_feature.seq_att_map:
          for key in seq_att_map.key:
            sequence_features.add(key)
          for hist_seq in seq_att_map.hist_seq:
            sequence_features.add(hist_seq)
    # compat with din
    for sequence_feature in config.model_config.seq_att_groups:
      for seq_att_map in sequence_feature.seq_att_map:
        for key in seq_att_map.key:
          sequence_features.add(key)
        for hist_seq in seq_att_map.hist_seq:
          sequence_features.add(hist_seq)
    excluded_features = excluded_features - sequence_features

    feature_configs = []
    for feature_config in config_util.get_compatible_feature_configs(config):
      feature_name = feature_config.feature_name if feature_config.HasField('feature_name') \
          else feature_config.input_names[0]
      if feature_name not in excluded_features:
        feature_configs.append(feature_config)

    if config.feature_configs:
      config.ClearField('feature_configs')
      config.feature_configs.extend(feature_configs)
    else:
      config.feature_config.ClearField('features')
      config.feature_config.features.extend(feature_configs)

    for feature_group in config.model_config.feature_groups:
      feature_names = []
      for feature_name in feature_group.feature_names:
        if feature_name not in excluded_features:
          feature_names.append(feature_name)
      feature_group.ClearField('feature_names')
      feature_group.feature_names.extend(feature_names)
    config_util.save_message(
        config,
        os.path.join(self._output_dir, os.path.basename(self._config_path)))

    if self._fg_path is not None and len(self._fg_path) > 0:
      with tf.gfile.Open(self._fg_path) as f:
        fg_json = json.load(f, object_pairs_hook=OrderedDict)
        features = []
        for feature in fg_json['features']:
          if 'feature_name' in feature:
            if feature['feature_name'] not in excluded_features:
              features.append(feature)
          else:
            features.append(feature)
        fg_json['features'] = features
      with tf.gfile.Open(
          os.path.join(self._output_dir, os.path.basename(self._fg_path)),
          'w') as f:
        json.dump(fg_json, f, indent=4)

  def _dump_to_csv(self, feature_importance, group_name):
    """Dump feature importance data to a csv file."""
    with tf.gfile.Open(
        os.path.join(self._output_dir,
                     'feature_dropout_ratio_%s.csv' % group_name), 'w') as f:
      df = pd.DataFrame(
          columns=['feature_name', 'mean_drop_p'],
          data=[list(kv) for kv in feature_importance.items()])
      df.to_csv(f, encoding='gbk')

  def _visualize_embedding_dim_importance(self, feature_dim_dropout_p):
    """Visualize embedding-wise importance visualization for every feature."""
    output_dir = os.path.join(self._output_dir, 'feature_dims_importance_pics')
    if not tf.gfile.Exists(output_dir):
      tf.gfile.MakeDirs(output_dir)

    plt.rcdefaults()
    for feature_name, feature_dropout_p in feature_dim_dropout_p.items():
      embedding_len = len(feature_dropout_p)
      embedding_dims = []
      for i in range(embedding_len):
        embedding_dims.append('dim_' + str(i + 1))
      y_pos = np.arange(len(embedding_dims))
      performance_list = []
      for i in range(0, embedding_len):
        performance_list.append(feature_dropout_p[i])
      fig, ax = plt.subplots()
      b = ax.barh(
          y_pos,
          performance_list,
          align='center',
          alpha=0.4,
          label='dropout_rate',
          lw=1)
      for rect in b:
        w = rect.get_width()
        ax.text(
            w,
            rect.get_y() + rect.get_height() / 2,
            '%.4f' % w,
            ha='left',
            va='center')
      plt.yticks(y_pos, embedding_dims)
      plt.xlabel(feature_name)
      plt.title('Dropout ratio')
      img_path = os.path.join(output_dir, feature_name + '.png')
      with tf.gfile.GFile(img_path, 'wb') as f:
        plt.savefig(f, format='png')

  def _visualize_feature_importance(self, feature_importance, group_name):
    """Draw feature importance histogram."""
    df = pd.DataFrame(
        columns=['feature_name', 'mean_drop_p'],
        data=[list(kv) for kv in feature_importance.items()])
    df['color'] = ['red' if x < 0.5 else 'green' for x in df['mean_drop_p']]
    df.sort_values('mean_drop_p', inplace=True, ascending=False)
    df.reset_index(inplace=True)
    # Draw plot
    plt.figure(figsize=(90, 200), dpi=100)
    plt.hlines(y=df.index, xmin=0, xmax=df.mean_drop_p)
    for x, y, tex in zip(df.mean_drop_p, df.index, df.mean_drop_p):
      plt.text(
          x,
          y,
          round(tex, 2),
          horizontalalignment='right' if x < 0 else 'left',
          verticalalignment='center',
          fontdict={
              'color': 'red' if x < 0 else 'green',
              'size': 14
          })
    # Decorations
    plt.yticks(df.index, df.feature_name, fontsize=20)
    plt.title('Dropout Ratio', fontdict={'size': 30})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    with tf.gfile.GFile(
        os.path.join(self._output_dir,
                     'feature_dropout_pic_%s.png' % group_name), 'wb') as f:
      plt.savefig(f, format='png')


if __name__ == '__main__':
  if FLAGS.model_type == 'variational_dropout':
    fs = VariationalDropoutFS(
        FLAGS.config_path,
        FLAGS.output_dir,
        FLAGS.topk,
        checkpoint_path=FLAGS.checkpoint_path,
        fg_path=FLAGS.fg_path,
        visualize=FLAGS.visualize)
    fs.process()
  else:
    raise ValueError('Unknown feature selection model type %s' %
                     FLAGS.model_type)
