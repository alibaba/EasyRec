import argparse
import datetime
import json
import os

import nni

from easy_rec.python.hpo_nni.pai_nni.code.metric_utils import get_result
from easy_rec.python.hpo_nni.pai_nni.code.pyodps_utils import create_odps
from easy_rec.python.hpo_nni.pai_nni.code.pyodps_utils import run_command
from easy_rec.python.hpo_nni.pai_nni.code.utils import parse_config
from easy_rec.python.hpo_nni.pai_nni.code.utils import set_value


def get_params():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--odps_config',
      type=str,
      help='odps_config.ini',
      default='../config/odps_config.ini')
  parser.add_argument(
      '--oss_config',
      type=str,
      help='excel config path',
      default='../config/.ossutilconfig')
  parser.add_argument(
      '--easyrec_cmd_config',
      type=str,
      help='pai config path',
      default='../config/easyrec_cmd_config_finetune')
  parser.add_argument('--exp_dir', type=str, help='exp dir', default='../exp')
  parser.add_argument(
      '--metric_config',
      type=str,
      help='metric config path',
      default='../config/metric_config')
  parser.add_argument(
      '--start_time',
      type=str,
      help='finetune start time',
      default='2022-05-30')
  parser.add_argument(
      '--end_time', type=str, help='finetune end time', default='2022-06-17')
  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':

  try:
    args = get_params()
    print('args:', args)

    odps_config = parse_config(args.odps_config)

    o = create_odps(
        access_id=odps_config['access_id'],
        access_key=odps_config['access_key'],
        project=odps_config['project_name'],
        endpoint=odps_config['end_point'])

    if args.oss_config is None:
      args.oss_config = os.path.join(os.environ['HOME'], '.ossutilconfig')

    # get parameters form tuner
    tuner_params = nni.get_next_parameter()

    # for tag
    experment_id = str(nni.get_experiment_id())
    trial_id = str(nni.get_trial_id())

    # for early stop,kill mc instance
    set_value('access_id', odps_config['access_id'], trial_id=trial_id)
    set_value('access_key', odps_config['access_key'], trial_id=trial_id)
    set_value('project', odps_config['project_name'], trial_id=trial_id)
    set_value('endpoint', odps_config['end_point'], trial_id=trial_id)

    datestart = datetime.datetime.strptime(args.start_time, '%Y-%m-%d')
    dateend = datetime.datetime.strptime(args.end_time, '%Y-%m-%d')

    sum = 0
    cnt = 0
    while (datestart <= dateend):
      print(datestart.strftime('%Y%m%d'))
      easyrec_cmd_config = parse_config(args.easyrec_cmd_config)

      # update parameter
      pre_edit = eval(easyrec_cmd_config.get('-Dedit_config_json', '{}'))
      pre_edit.update(tuner_params)
      if cnt > 0:
        pre_edit['train_config.fine_tune_checkpoint'] = os.path.join(
            pre_edit['train_config.fine_tune_checkpoint'],
            experment_id + '_' + trial_id)
      edit_json = json.dumps(pre_edit)
      easyrec_cmd_config['-Dedit_config_json'] = edit_json
      print(easyrec_cmd_config['-Dedit_config_json'])
      easyrec_cmd_config['-Dtrain_tables'] = easyrec_cmd_config[
          '-Dtrain_tables'].replace('{bizdate}', datestart.strftime('%Y%m%d'))

      next_day = datestart + datetime.timedelta(days=1)
      easyrec_cmd_config['-Deval_tables'] = easyrec_cmd_config[
          '-Deval_tables'].replace('{eval_ymd}', next_day.strftime('%Y%m%d'))

      pre_day = datestart - datetime.timedelta(days=1)
      easyrec_cmd_config['-Dedit_config_json'] = easyrec_cmd_config[
          '-Dedit_config_json'].replace('{predate}', pre_day.strftime('%Y%m%d'))

      easyrec_cmd_config['-Dmodel_dir'] = easyrec_cmd_config[
          '-Dmodel_dir'].replace('{bizdate}', datestart.strftime('%Y%m%d'))

      easyrec_cmd_config['-Dmodel_dir'] = os.path.join(
          easyrec_cmd_config['-Dmodel_dir'], experment_id + '_' + trial_id)
      print(easyrec_cmd_config)

      # trial id for early stop
      run_command(o, easyrec_cmd_config, trial_id)

      filepath = os.path.join(easyrec_cmd_config['-Dmodel_dir'], 'eval_val/')
      dst_filepath = os.path.join(args.exp_dir, experment_id + '_' + trial_id,
                                  datestart.strftime('%Y%m%d'))
      metric_dict = parse_config(args.metric_config)
      print('filepath:', filepath)
      print('dst_file_path:', dst_filepath)
      print('metric dict:', metric_dict)
      best_res, best_event = get_result(
          filepath,
          dst_filepath,
          metric_dict,
          trial_id,
          oss_config=args.oss_config,
          nni_report=False)
      if best_res:
        nni.report_intermediate_result(best_res)
        sum += best_res
      cnt += 1
      datestart += datetime.timedelta(days=1)

    if cnt > 0:
      nni.report_final_result(sum / cnt)

  except Exception as exception:
    print('exception: ', exception)
    raise
