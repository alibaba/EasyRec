import json
import os
import pathlib

filepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

outfile_dir = os.path.join(filepath, 'exp_json')


def get_filepath(trial_id=None):
  pathlib.Path(outfile_dir).mkdir(parents=True, exist_ok=True)
  if trial_id:
    outfile = os.path.join(outfile_dir, str(trial_id) + '_mc.json')
  else:
    outfile = os.path.join(outfile_dir, 'mc.json')

  if not os.path.exists(outfile):
    with open(outfile, 'w') as f:
      json.dump({}, f)
  return outfile


def set_value(key, value, trial_id=None):
  outfile = get_filepath(trial_id=trial_id)
  with open(outfile, 'r') as f:
    _global_dict = json.load(f)
  _global_dict[key] = value
  with open(outfile, 'w') as f:
    json.dump(_global_dict, f)


def get_value(key, defValue=None, trial_id=None):
  outfile = get_filepath(trial_id=trial_id)
  with open(outfile, 'r') as f:
    _global_dict = json.load(f)
  print('dict:', _global_dict)
  try:
    return _global_dict[key]
  except KeyError:
    return defValue


def try_parse(v):
  try:
    return int(v)
  except ValueError:
    try:
      return float(v)
    except ValueError:
      return v


def parse_config(config_path):
  assert os.path.exists(config_path)
  config = {}
  with open(config_path, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      if len(line_str) == 0:
        continue
      if line_str[0] == '#':
        continue
      if '=' in line_str:
        tmp_id = line_str.find('=')
        key = line_str[:tmp_id].strip()
        val = try_parse(line_str[(tmp_id + 1):].strip())
        config[key] = val
  return config
