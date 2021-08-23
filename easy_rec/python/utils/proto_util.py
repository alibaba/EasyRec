# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging


def copy_obj(proto_obj):
  """Make a copy of proto_obj so that later modifications of tmp_obj will have no impact on proto_obj.

  Args:
    proto_obj: a protobuf message
  Return:
    a copy of proto_obj
  """
  tmp_obj = type(proto_obj)()
  tmp_obj.CopyFrom(proto_obj)
  return tmp_obj


def get_norm_embed_name(name, verbose=False):
  """For embedding export to redis.

  Args:
    name: variable name
    verbose: whether to dump the embed_names
  Return:
    embedding_name: normalized embedding_name
    embedding_part_id: normalized embedding part_id
    if embedding_weights not in name, return None, None
  """
  name_toks = name.split('/')
  for i in range(0, len(name_toks) - 1):
    if name_toks[i + 1].startswith('embedding_weights:'):
      var_id = name_toks[i + 1].replace('embedding_weights:', '')
      tmp_name = '/'.join(name_toks[:i + 1])
      if var_id != '0':
        tmp_name = tmp_name + '_' + var_id
      if verbose:
        logging.info('norm %s to %s' % (name, tmp_name))
      return tmp_name, 0
    if i > 1 and name_toks[i + 1].startswith('part_') and \
       name_toks[i] == 'embedding_weights':
      tmp_name = '/'.join(name_toks[:i])
      part_id = name_toks[i + 1].replace('part_', '')
      part_toks = part_id.split(':')
      if len(part_toks) >= 2 and part_toks[1] != '0':
        tmp_name = tmp_name + '_' + part_toks[1]
      if verbose:
        logging.info('norm %s to %s' % (name, tmp_name))
      return tmp_name, int(part_toks[0])

  # input_layer/app_category_embedding/app_category_embedding_weights/SparseReshape
  # => input_layer/app_category_embedding
  for i in range(0, len(name_toks) - 1):
    if name_toks[i + 1].endswith('_embedding_weights'):
      tmp_name = '/'.join(name_toks[:i + 1])
      if verbose:
        logging.info('norm %s to %s' % (name, tmp_name))
      return tmp_name, 0
  # input_layer/app_category_embedding/embedding_weights
  # => input_layer/app_category_embedding
  for i in range(0, len(name_toks) - 1):
    if name_toks[i + 1] == 'embedding_weights':
      tmp_name = '/'.join(name_toks[:i + 1])
      if verbose:
        logging.info('norm %s to %s' % (name, tmp_name))
      return tmp_name, 0
  logging.warning('Failed to norm: %s' % name)
  return None, None
