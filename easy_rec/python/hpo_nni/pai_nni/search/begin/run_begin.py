import argparse
import json
import logging
import os

import nni

from easy_rec.python.hpo_nni.pai_nni.core.metric_utils import report_result
from easy_rec.python.hpo_nni.pai_nni.core.pyodps_utils import create_odps
from easy_rec.python.hpo_nni.pai_nni.core.pyodps_utils import run_command
from easy_rec.python.hpo_nni.pai_nni.core.utils import parse_ini
from easy_rec.python.hpo_nni.pai_nni.core.utils import set_value


def get_params():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config', type=str, help='config path', default='./config_begin.ini')
  parser.add_argument('--exp_dir', type=str, help='exp dir', default='../exp')
  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':

  try:
    args = get_params()
    print('args:', args)
    config = parse_ini(args.config)

    metric_dict = config['metric_config']
    print('metric dict:', metric_dict)

    odps_config = config['odps_config']
    print('odps_config:', odps_config)

    oss_config = config['oss_config']

    easyrec_cmd_config = config['easyrec_cmd_config']
    print('easyrec_cmd_config:', easyrec_cmd_config)

    o = create_odps(
        access_id=oss_config['accessKeyID'],
        access_key=oss_config['accessKeySecret'],
        project=odps_config['project_name'],
        endpoint=odps_config['odps_endpoint'])

    # get parameters form tuner
    tuner_params = nni.get_next_parameter()

    # for tag
    experment_id = str(nni.get_experiment_id())
    trial_id = str(nni.get_trial_id())

    # for early stop,kill mc instance
    set_value('access_id', oss_config['accessKeyID'], trial_id=trial_id)
    set_value('access_key', oss_config['accessKeySecret'], trial_id=trial_id)
    set_value('project', odps_config['project_name'], trial_id=trial_id)
    set_value('endpoint', odps_config['odps_endpoint'], trial_id=trial_id)

    # update parameter
    pre_edit = eval(easyrec_cmd_config.get('-Dedit_config_json', '{}'))
    pre_edit.update(tuner_params)
    edit_json = json.dumps(pre_edit)
    easyrec_cmd_config['-Dedit_config_json'] = edit_json
    print('-Dedit_config_json:', easyrec_cmd_config['-Dedit_config_json'])

    # report metric
    easyrec_cmd_config['-Dmodel_dir'] = os.path.join(
        easyrec_cmd_config['-Dmodel_dir'], experment_id + '_' + trial_id)
    filepath = os.path.join(easyrec_cmd_config['-Dmodel_dir'], 'eval_val/')
    dst_filepath = os.path.join(args.exp_dir, experment_id + '_' + trial_id)
    print('filepath:', filepath)
    print('dst_file_path:', dst_filepath)
    report_result(
        filepath, dst_filepath, metric_dict, trial_id, oss_config=oss_config)

    # trial id for early stop
    run_command(o, easyrec_cmd_config, trial_id)

    # kill the report_result
    set_value(trial_id + '_exit', '1', trial_id=trial_id)

  except Exception:
    logging.exception('run begin error')
    set_value(trial_id + '_exit', '1', trial_id=trial_id)
    exit(1)
