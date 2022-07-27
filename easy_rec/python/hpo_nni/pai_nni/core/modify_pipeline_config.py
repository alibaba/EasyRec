import argparse
import os

from easy_rec.python.hpo_nni.pai_nni.core.metric_utils import copy_file
from easy_rec.python.hpo_nni.pai_nni.core.metric_utils import upload_file
from easy_rec.python.hpo_nni.pai_nni.core.utils import parse_config
from easy_rec.python.utils import config_util


def get_params():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--pipeline_config_path',
      type=str,
      help='pipeline config path',
      default='../../../../../samples/hpo/pipeline.config')
  parser.add_argument(
      '--save_path',
      type=str,
      help='modify pipeline config path',
      default='../../../../../samples/hpo/pipeline_finetune.config')
  parser.add_argument(
      '--learning_rate',
      type=float,
      help='easyrec cmd config path',
      default=1e-6)
  parser.add_argument(
      '--oss_config',
      type=str,
      help='excel config path',
      default=os.path.join(os.path.expanduser('~'), '.ossutilconfig'))
  args, _ = parser.parse_known_args()
  return args


def modify_config(args):
  if args.pipeline_config_path.startswith('oss://'):
    oss_config = parse_config(args.oss_config)
    print('pipeline_config_path:', args.pipeline_config_path)
    copy_file(args.pipeline_config_path, './temp.config', oss_config=oss_config)
    pipeline_config_path = './temp.config'
  else:
    pipeline_config_path = args.pipeline_config_path

  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  optimizer_configs = pipeline_config.train_config.optimizer_config
  for optimizer_config in optimizer_configs:
    optimizer = optimizer_config.WhichOneof('optimizer')
    optimizer = getattr(optimizer_config, optimizer)
    learning_rate = optimizer.learning_rate.WhichOneof('learning_rate')
    learning_rate = getattr(optimizer.learning_rate, learning_rate)
    if hasattr(learning_rate, 'learning_rate'):
      learning_rate.learning_rate = args.learning_rate

    elif hasattr(learning_rate, 'initial_learning_rate'):
      if hasattr(learning_rate, 'min_learning_rate'):
        if args.learning_rate < learning_rate.min_learning_rate:
          print('args.learning_rate {} < learning_rate.min_learning_rate {}, '
                'we will use the learning_rate.min_learning_rate {}'.format(
                    args.learning_rate, learning_rate.min_learning_rate,
                    learning_rate.min_learning_rate))
        learning_rate.initial_learning_rate = max(
            args.learning_rate, learning_rate.min_learning_rate)
      else:
        learning_rate.initial_learning_rate = args.learning_rate

    elif hasattr(learning_rate, 'learning_rate_base'):

      if hasattr(learning_rate, 'end_learning_rate'):
        if args.learning_rate < learning_rate.end_learning_rate:
          print('args.learning_rate {} < learning_rate.end_learning_rate {},'
                ' we will use the learning_rate.end_learning_rate {}'.format(
                    args.learning_rate, learning_rate.end_learning_rate,
                    learning_rate.end_learning_rate))
        learning_rate.learning_rate_base = max(args.learning_rate,
                                               learning_rate.end_learning_rate)
      else:
        learning_rate.learning_rate_base = args.learning_rate

  if args.save_path.startswith('oss://'):
    save_path = './pipeline_finetune.config'
  else:
    save_path = args.save_path

  save_dir = os.path.dirname(save_path)
  file_name = os.path.basename(save_path)
  config_util.save_pipeline_config(
      pipeline_config=pipeline_config, directory=save_dir, filename=file_name)

  if args.save_path.startswith('oss://'):
    oss_config = parse_config(args.oss_config)
    upload_file(
        args.save_path, './pipeline_finetune.config', oss_config=oss_config)
    os.remove(save_path)

  if args.pipeline_config_path.startswith('oss://'):
    os.remove(pipeline_config_path)


def get_learning_rate(args):

  if args.save_path.startswith('oss://'):
    oss_config = parse_config(args.oss_config)
    print('pipeline_config_path:', args.save_path)
    copy_file(args.save_path, './temp.config', oss_config=oss_config)
    pipeline_config_path = './temp.config'
  else:
    pipeline_config_path = args.save_path

  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  if args.save_path.startswith('oss://'):
    os.remove(pipeline_config_path)
  optimizer_configs = pipeline_config.train_config.optimizer_config
  for optimizer_config in optimizer_configs:
    optimizer = optimizer_config.WhichOneof('optimizer')
    optimizer = getattr(optimizer_config, optimizer)
    learning_rate = optimizer.learning_rate.WhichOneof('learning_rate')
    learning_rate = getattr(optimizer.learning_rate, learning_rate)
    if hasattr(learning_rate, 'learning_rate'):
      return learning_rate.learning_rate

    elif hasattr(learning_rate, 'initial_learning_rate'):
      return learning_rate.initial_learning_rate

    elif hasattr(learning_rate, 'learning_rate_base'):
      return learning_rate.learning_rate_base
  return None


if __name__ == '__main__':
  args = get_params()
  print('args:', args)
  modify_config(args)
  print('final learning_rate:', get_learning_rate(args))
