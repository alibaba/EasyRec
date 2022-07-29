# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import logging

import odps
from hpo_nni.core.utils import set_value
from hpo_nni.core.utils import try_parse

Command = collections.namedtuple('Command', ['name', 'project', 'parameters'])


def create_odps(project, access_id, access_key, endpoint, biz_id=None):

  def init():
    if biz_id:
      # ref: http://pyodps.alibaba.net/pyodps-docs/en/latest/base-sql.html?highlight=biz_id#biz-id
      odps.options.biz_id = biz_id

    o = odps.ODPS(
        access_id=access_id,
        secret_access_key=access_key,
        endpoint=endpoint,
        project=project)

    proj = o.get_project(project)
    if not o.exist_project(proj):
      raise ValueError('ODPS init failed, please check your project name.')
    return o

  return init()


def parse_easyrec_cmd_config(easyrec_cmd_config):
  """When val='x', convert "'x'"->'x' when val="x",convert '"x"'->'x'."""
  name = easyrec_cmd_config['-name']
  project = easyrec_cmd_config['-project']

  params = {}
  for k, val in easyrec_cmd_config.items():
    if k.startswith('-D'):
      if val[0] == "'" and val[-1] == "'":
        val = val[1:-1]
      if val[0] == '"' and val[-1] == '"':
        val = val[1:-1]
      params[k.replace('-D', '')] = try_parse(val)
  return Command(name=name, project=project, parameters=params)


def run_command(o, easyrec_cmd_config, trial_id=None):
  # parse command
  command = parse_easyrec_cmd_config(easyrec_cmd_config=easyrec_cmd_config)
  logging.info('command %s', command)
  instance = o.run_xflow(
      xflow_name=command.name,
      xflow_project=command.project,
      parameters=command.parameters)
  for inst_name, inst in o.iter_xflow_sub_instances(instance):
    logging.info('inst name: %s', inst_name)
    logging.info(inst.get_logview_address())
    logging.info('instance id%s', inst)

    if inst_name == 'train' and trial_id:
      set_value(trial_id, str(inst), trial_id=trial_id)

  # for report result
  if trial_id:
    set_value(trial_id + '_exit', '1', trial_id=trial_id)