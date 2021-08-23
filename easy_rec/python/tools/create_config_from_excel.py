# -*-encoding:utf-8-*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import math
import sys

import numpy as np
import pandas as pd

from easy_rec.python.utils import config_util

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')


class ModelConfigConverter:

  def __init__(self, excel_path, output_path, model_type, column_separator,
               incol_separator, train_input_path, eval_input_path, model_dir):
    self._excel_path = excel_path
    self._output_path = output_path
    self._model_type = model_type
    self._column_separator = column_separator
    self._incol_separator = incol_separator
    self._dict_global = self._parse_global()
    self._tower_dicts = {}
    self._feature_names = []
    self._feature_details = {}
    self._label = ''
    self._train_input_path = train_input_path
    self._eval_input_path = eval_input_path
    self._model_dir = model_dir
    if not self._model_dir:
      self._model_dir = 'experiments/demo'
      logging.warning('model_dir is not specified, set to %s' % self._model_dir)

  def _get_type_name(self, input_name):
    type_dict = {
        'bigint': 'INT64',
        'double': 'DOUBLE',
        'float': 'FLOAT',
        'string': 'STRING',
        'bool': 'BOOL'
    }
    return type_dict[input_name]

  def _get_type_default(self, input_name):
    type_dict = {
        'bigint': '0',
        'double': '0.0',
        'float': '0.0',
        'string': '',
        'bool': 'false'
    }
    return type_dict[input_name]

  def _parse_global(self):
    df = pd.read_excel(self._excel_path, sheet_name='global')
    dict_global = {}
    for i, row in df.iterrows():
      field = {}
      name = field['name'] = row['name'].strip()
      field['type_name'] = row['type']
      field['hash_bucket_size'] = row['hash_bucket_size']
      field['embedding_dim'] = row['embedding_dim']
      field['default_value'] = row['default_value']
      dict_global[name] = field
    return dict_global

  def _add_to_tower(self, tower_name, field):
    if tower_name.lower() == 'nan':
      return
    if tower_name != 'label':
      if self._model_type == 'deepfm':
        if tower_name == 'deep':
          tower_names = ['deep']
        elif tower_name == 'wide':
          tower_names = ['wide']
        elif tower_name == 'wide_and_deep':
          tower_names = ['wide', 'deep']
        else:
          raise ValueError(
              'invalid tower_name[%s] for deepfm model, '
              'only[label, deep, wide, wide_and_deep are supported]' %
              tower_name)
        for tower_name in tower_names:
          if tower_name in self._tower_dicts:
            self._tower_dicts[tower_name].append(field)
          else:
            self._tower_dicts[tower_name] = [field]
      else:
        if tower_name in self._tower_dicts:
          self._tower_dicts[tower_name].append(field)
        else:
          self._tower_dicts[tower_name] = [field]

  def _is_str(self, v):
    if isinstance(v, str):
      return True
    try:
      if isinstance(v, unicode):  # noqa: F821
        return True
    except NameError:
      return False
    return False

  def _parse_features(self):
    df = pd.read_excel(self._excel_path, sheet_name='features')
    for i, row in df.iterrows():
      field = {}
      name = field['name'] = row['name'].strip()
      self._feature_names.append(name)
      field['data_type'] = row['data_type'].strip()
      field['type'] = row['type'].strip()
      g = str(row['global']).strip()

      if g and g != 'nan':
        field['global'] = g

      field['field_name'] = name

      if row['type'].strip() == 'label':
        self._label = name

      if 'global' in field and field['global'] in self._dict_global:
        # 如果是global 有值，就跳过
        def _is_good(v):
          return str(v) not in ['nan', '']

        if _is_good(self._dict_global[field['global']]['default_value']):
          field['default_value'] = self._dict_global[
              field['global']]['default_value']
        if _is_good(self._dict_global[field['global']]['hash_bucket_size']):
          field['hash_bucket_size'] = self._dict_global[
              field['global']]['hash_bucket_size']
        if _is_good(self._dict_global[field['global']]['embedding_dim']):
          field['embedding_dim'] = self._dict_global[
              field['global']]['embedding_dim']
        field['embedding_name'] = field['global']

      for t in [
          'type', 'global', 'hash_bucket_size', 'embedding_dim',
          'default_value', 'weights', 'boundaries'
      ]:
        if t not in row:
          continue
        v = row[t]
        if v not in ['', ' ', 'NaN', np.NaN, np.NAN, 'nan']:
          if self._is_str(v):
            field[t] = v.strip()
          elif not math.isnan(v):
            field[t] = int(v)

        if t == 'default_value' and t not in field:
          field[t] = ''
          if field['type'] == 'dense':
            field[t] = 0.0

      if field['type'] == 'weights':
        field['default_value'] = '1'

      tower_name = row['group']
      if name in self._dict_global:
        field['type'] = 'category'
        field['hash_bucket_size'] = self._dict_global[name]['hash_bucket_size']
        field['embedding_dim'] = self._dict_global[name]['embedding_dim']
        field['default_value'] = self._dict_global[name]['default_value']

      if field['data_type'] == 'bigint':
        field['default_value'] = 0
      elif field['data_type'] == 'double':
        field['default_value'] = 0.0

      if field['type'] not in ['notneed', 'not_need', 'not_needed']:
        tower_name = str(tower_name).strip()
        self._add_to_tower(tower_name, field)
      self._feature_details[name] = field

    # check that tag features weights are one of the fields
    for name, config in self._feature_details.items():
      if config['type'] == 'tags':
        if 'weights' in config and config[
            'weights'] not in self._feature_details:
          raise ValueError(config['weights'] + ' not in field names')

  def _write_train_eval_config(self, fout):
    fout.write('train_input_path: "%s"\n' % self._train_input_path)
    fout.write('eval_input_path: "%s"\n' % self._eval_input_path)
    fout.write("""
    model_dir: "%s"

    train_config {
      log_step_count_steps: 200
      # fine_tune_checkpoint: ""
      optimizer_config: {
        adam_optimizer: {
          learning_rate: {
            exponential_decay_learning_rate {
              initial_learning_rate: 0.0001
              decay_steps: 10000
              decay_factor: 0.5
              min_learning_rate: 0.0000001
            }
          }
        }
      }
      num_steps: 2000
      sync_replicas: true
    }

    eval_config {
      metrics_set: {
           auc {}
      }
    }""" % self._model_dir)

  def _write_deepfm_config(self, fout):
    # write model_config
    fout.write('model_config:{\n')
    fout.write('  model_class: "DeepFM"\n')

    # write feature group configs
    tower_names = list(self._tower_dicts.keys())
    tower_names.sort()
    for tower_name in tower_names:
      fout.write('  feature_groups: {\n')
      fout.write('    group_name: "%s"\n' % tower_name)
      curr_feas = self._tower_dicts[tower_name]
      for fea in curr_feas:
        if fea['type'] == 'weights':
          continue
        fout.write('    feature_names: "%s"\n' % fea['name'])
      fout.write('    wide_deep:%s\n' % tower_name.upper())
      fout.write('  }\n')

    # write deepfm configs
    fout.write("""
      deepfm {
        dnn {
          hidden_units: [128, 64, 32]
        }
        final_dnn {
          hidden_units: [128, 64]
        }
        wide_output_dim: 16
        l2_regularization: 1e-5
      }
      embedding_regularization: 1e-5
    }
    """)

  def _write_multi_tower_config(self, fout):
    # write model_config
    fout.write('model_config:{\n')
    fout.write('  model_class: "MultiTower"\n')

    # write each tower features
    tower_names = list(self._tower_dicts.keys())
    tower_names.sort()
    for tower_name in tower_names:
      fout.write('  feature_groups: {\n')
      fout.write('    group_name: "%s"\n' % tower_name)
      curr_feas = self._tower_dicts[tower_name]
      for fea in curr_feas:
        if fea['type'] == 'weights':
          continue
        fout.write('    feature_names: "%s"\n' % fea['name'])
      fout.write('    wide_deep:DEEP\n')
      fout.write('  }\n')

    # write each tower dnn configs
    fout.write('multi_tower { \n')

    for tower_name in tower_names:
      fout.write("""
      towers {
        input: "%s"
        dnn {
          hidden_units: [256, 192, 128]
        }
      }""" % tower_name)

    fout.write("""
        final_dnn {
          hidden_units: [192, 128, 64]
        }
        l2_regularization: 1e-5
      }
      embedding_regularization: 1e-5
    }""")

  def _write_data_config(self, fout):
    fout.write('data_config {\n')
    fout.write('  separator: "%s"\n' % self._column_separator)
    for name in self._feature_names:
      fout.write('  input_fields: {\n')
      fout.write('    input_name: "%s"\n' % name)
      fout.write('    input_type: %s\n' %
                 self._get_type_name(self._feature_details[name]['data_type']))
      if 'default_value' in self._feature_details[name]:
        fout.write('    default_val:"%s"\n' %
                   self._feature_details[name]['default_value'])
      fout.write('  }\n')

    fout.write('  label_fields: "%s"\n' % self._label)
    fout.write("""
      batch_size: 1024
      prefetch_size: 32
      input_type: CSVInput
    }""")

  def _write_feature_config(self, fout):
    for name in self._feature_names:
      feature = self._feature_details[name]
      if feature['type'] in ['weights', 'notneed', 'label']:
        continue
      if name == self._label:
        continue
      fout.write('feature_configs: {\n')
      fout.write('  input_names: "%s"\n' % name)
      if feature['type'] == 'category':
        fout.write('  feature_type: IdFeature\n')
        fout.write('  embedding_dim: %d\n' % feature['embedding_dim'])
        fout.write('  hash_bucket_size: %d\n' % feature['hash_bucket_size'])
        if 'embedding_name' in feature:
          fout.write('  embedding_name: "%s"\n' % feature['embedding_name'])
      elif feature['type'] == 'dense':
        fout.write('  feature_type: RawFeature\n')
        if self._model_type == 'deepfm':
          assert feature[
              'boundaries'] != '', 'raw features must be discretized by specifying boundaries'
        if 'boundaries' in feature and feature['boundaries'] != '':
          fout.write('  boundaries: [%s]\n' %
                     str(feature['boundaries']).strip())
          fout.write('  embedding_dim: %d\n' % int(feature['embedding_dim']))
      elif feature['type'] == 'tags':
        if 'weights' in feature:
          fout.write('  input_names: "%s"\n' % feature['weights'])
        fout.write('  feature_type: TagFeature\n')
        fout.write('  hash_bucket_size: %d\n' % feature['hash_bucket_size'])
        fout.write('  embedding_dim: %d\n' % feature['embedding_dim'])
        if 'embedding_name' in feature:
          fout.write('  embedding_name: "%s"\n' % feature['embedding_name'])
        fout.write('  separator: "%s"\n' % self._incol_separator)
      elif feature['type'] == 'indexes':
        fout.write('  feature_type: TagFeature\n')
        assert 'hash_bucket_size' in feature
        fout.write('  num_buckets: %d\n' % feature['hash_bucket_size'])
        if 'embedding_dim' in feature:
          fout.write('  embedding_dim: %d\n' % feature['embedding_dim'])
        if 'embedding_name' in feature:
          fout.write('  embedding_name: "%s"\n' % feature['embedding_name'])
        fout.write('  separator: "%s"\n' % self._incol_separator)
      else:
        assert False, 'invalid feature types: %s' % feature['type']
      fout.write('}\n')

  def convert(self):
    self._parse_features()
    logging.info(
        'TOWERS[%d]: %s' %
        (len(self._tower_dicts), ','.join(list(self._tower_dicts.keys()))))
    with open(self._output_path, 'w') as fout:
      self._write_train_eval_config(fout)
      self._write_data_config(fout)
      self._write_feature_config(fout)
      if self._model_type == 'deepfm':
        self._write_deepfm_config(fout)
      elif self._model_type == 'multi_tower':
        self._write_multi_tower_config(fout)
      else:
        logging.warning(
            'the model_config could not be generated automatically, you have to write the model_config manually.'
        )
    # reformat the config
    pipeline_config = config_util.get_configs_from_pipeline_file(
        self._output_path)
    config_util.save_message(pipeline_config, self._output_path)


model_types = ['deepfm', 'multi_tower']

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_type',
      type=str,
      choices=model_types,
      help='model type, currently support: %s' % ','.join(model_types))
  parser.add_argument('--excel_path', type=str, help='excel config path')
  parser.add_argument('--output_path', type=str, help='generated config path')
  parser.add_argument(
      '--column_separator',
      type=str,
      default=',',
      help='column separator, separator betwen features')
  parser.add_argument(
      '--incol_separator',
      type=str,
      default='|',
      help='separator within features, such as tag features')
  parser.add_argument(
      '--train_input_path', type=str, default='', help='train input path')
  parser.add_argument(
      '--eval_input_path', type=str, default='', help='eval input path')
  parser.add_argument('--model_dir', type=str, default='', help='model dir')
  args = parser.parse_args()

  if not args.excel_path or not args.output_path:
    parser.print_usage()
    sys.exit(1)

  logging.info('column_separator = %s in_column_separator = %s' %
               (args.column_separator, args.incol_separator))

  converter = ModelConfigConverter(args.excel_path, args.output_path,
                                   args.model_type, args.column_separator,
                                   args.incol_separator, args.train_input_path,
                                   args.eval_input_path, args.model_dir)
  converter.convert()
  logging.info('Conversion done')
  logging.info('Tips:')
  if args.train_input_path == '' or args.eval_input_path == '':
    logging.info('*.you have to update train_input_path,  eval_input_path')
  logging.info('*.you may need to adjust dnn config or final_dnn config')
