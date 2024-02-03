# -*- encoding:utf-8 -*-
# two config files are used: .git_bin_path  .git_bin_url
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import traceback

blank_split = re.compile('[\t ]')

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s[%(lineno)d] : %(message)s',
    level=logging.INFO)

try:
  import oss2
except ImportError:
  logging.error(
      'please install python_oss from https://github.com/aliyun/aliyun-oss-python-sdk.git'
  )
  sys.exit(1)

git_bin_path = '.git_bin_path'
git_bin_url_path = '.git_bin_url'
# temporary storage path
git_oss_cache_dir = '.git_oss_cache'


# get project name by using git remote -v
def get_proj_name():
  proj_name = subprocess.check_output(['git', 'remote', '-v'])
  proj_name = proj_name.decode('utf-8')
  proj_name = proj_name.split('\n')[0]
  proj_name = blank_split.split(proj_name)[1]
  proj_name = proj_name.split('/')[-1]
  proj_name = proj_name.replace('.git', '')
  return proj_name


# load .git_bin_url
# local_path md5 remote_path
def load_git_url():
  git_bin_url_map = {}
  try:
    with open(git_bin_url_path) as fin:
      for line_str in fin:
        line_str = line_str.strip()
        line_json = json.loads(line_str)
        git_bin_url_map[line_json['leaf_path']] = (line_json['sig'],
                                                   line_json['remote_path'])
  except Exception as ex:
    logging.warning('exception: %s' % str(ex))
    pass
  return git_bin_url_map


def save_git_url(git_bin_url_map):
  with open(git_bin_url_path, 'w') as fout:
    keys = list(git_bin_url_map.keys())
    keys.sort()
    for key in keys:
      val = git_bin_url_map[key]
      tmp_str = '{"leaf_path": "%s", "sig": "%s", "remote_path": "%s"}' % (
          key, val[0], val[1])
      fout.write('%s\n' % tmp_str)


def path2name(path):
  name = path.replace('//', '/')
  name = path.replace('/', '_')
  if name[-1] == '_':
    return name[:-1]
  elif name == '.':
    return 'curr_dir'
  else:
    return name


def get_file_arr(path):
  archive_files = []
  if os.path.isdir(path):
    for one_file in os.listdir(path):
      one_path = path + '/' + one_file
      if not os.path.isdir(one_path):
        archive_files.append(one_path)
    return archive_files
  else:  # just a file
    archive_files.append(path)
  return archive_files


def load_git_bin():
  file_arr = {}
  if not os.path.exists(git_bin_path):
    return file_arr

  with open(git_bin_path, 'r') as fin:
    for line_str in fin:
      line_str = line_str.strip()
      try:
        line_json = json.loads(line_str)
        file_arr[line_json['leaf_name']] = line_json['leaf_file']
      except Exception as ex:
        logging.warning('%s is corrupted : %s' %
                        (git_bin_path, traceback.format_exc(ex)))
  return file_arr


def save_git_bin(git_arr):
  leaf_paths = list(git_arr.keys())
  leaf_paths.sort()
  with open(git_bin_path, 'w') as fout:
    for leaf_path in leaf_paths:
      leaf_files = git_arr[leaf_path]
      leaf_files.sort()
      # make sure that leaf_name is in front of leaf_file
      tmp_str = '{"leaf_name": "%s", "leaf_file": %s}' % (
          leaf_path, json.dumps(leaf_files))
      fout.write('%s\n' % tmp_str)


def recheck_git_bin():
  file_arr = load_git_bin()
  update = False
  del_arr = []
  for leaf_path in file_arr:
    leaf_files = file_arr[leaf_path]
    good_leaf_files = [x for x in leaf_files if os.path.exists(x)]
    if not os.path.exists(leaf_path):
      del_arr.append(leaf_path)
      update = True
    elif len(good_leaf_files) != len(leaf_files):
      file_arr[leaf_path] = good_leaf_files
      update = True
  for leaf_path in del_arr:
    del file_arr[leaf_path]
  if update:
    save_git_bin(file_arr)
  return file_arr


# check whether a folder changes by check md5 of the tar file of the folder
# note -z option is not used, because the file has random effects
# the md5files are saved in .git_bin_url
def get_local_sig(leaf_files):
  if len(leaf_files) == 0:
    logging.warning('no leaf files')
    return None
  leaf_files = sorted(leaf_files)
  m = hashlib.md5()
  block_size = 1024 * 1024 * 8
  for one_file in leaf_files:
    with open(one_file, 'rb') as fin:
      for chunk in iter(lambda: fin.read(block_size), b''):
        m.update(chunk)
  return m.hexdigest()


def list_leafs(curr_path):
  bottom_dir = []
  if os.path.isdir(curr_path):
    for root, dirs, files in os.walk(curr_path, topdown=True):
      if len(dirs) == 0 or len(files) > 0:
        if root[-1] == '/':
          root = root[:-1]
        file_arr = get_file_arr(root)
        bottom_dir.append((root, file_arr))
  else:  # a single file
    curr_dir = os.path.dirname(curr_path)
    if curr_dir == '':
      curr_dir = '.'
    bottom_dir.append((curr_dir, [curr_path]))
  return bottom_dir


# check whether lst0 and lst1 contain the same string elements
def lst_eq(lst0, lst1):
  if len(lst0) != len(lst1):
    return False
  for x in lst1:
    if x not in lst0:
      return False
  return True


def merge_lst(lst0, lst1):
  for a in lst1:
    if a not in lst0:
      lst0.append(a)
  return lst0


def has_conflict(leaf_path, leaf_files):
  if not os.path.exists(leaf_path):
    return False
  for leaf_file in leaf_files:
    if os.path.exists(leaf_file):
      return True
  return False


def get_yes_no(msg):
  while True:
    logging.info(msg)
    tmp_op = sys.stdin.readline()
    tmp_op = tmp_op.strip()
    if len(tmp_op) == 0:
      break
    elif tmp_op[0] == 'Y' or tmp_op[0] == 'y':
      update = True
      break
    elif tmp_op[0] == 'N' or tmp_op[0] == 'n':
      update = False
      break
  return update


if __name__ == '__main__':
  if len(sys.argv) < 2:
    logging.error(
        'usage: python git_lfs.py [pull] [push] [add filename] [resolve_conflict]'
    )
    sys.exit(1)
  with open('.git_oss_config_pub', 'r') as fin:
    git_oss_data_dir = None
    host = None
    bucket_name = None
    git_oss_private_path = None
    enable_accelerate = 0
    accl_endpoint = None
    for line_str in fin:
      line_str = line_str.strip()
      if len(line_str) == 0:
        continue
      if line_str.startswith('#'):
        continue
      line_str = line_str.replace('~/', os.environ['HOME'] + '/')
      line_str = line_str.replace('${TMPDIR}/',
                                  os.environ.get('TMPDIR', '/tmp/'))
      line_str = line_str.replace('${PROJECT_NAME}', get_proj_name())
      line_tok = [x.strip() for x in line_str.split('=') if x != '']
      if line_tok[0] == 'host':
        host = line_tok[1]
      elif line_tok[0] == 'git_oss_data_dir':
        git_oss_data_dir = line_tok[1].strip('/')
      elif line_tok[0] == 'bucket_name':
        bucket_name = line_tok[1]
      elif line_tok[0] == 'git_oss_private_config':
        git_oss_private_path = line_tok[1]
        if git_oss_private_path.startswith('~/'):
          git_oss_private_path = os.path.join(os.environ['HOME'],
                                              git_oss_private_path[2:])
      elif line_tok[0] == 'git_oss_cache_dir':
        git_oss_cache_dir = line_tok[1]
      elif line_tok[0] == 'accl_endpoint':
        accl_endpoint = line_tok[1]

    logging.info('git_oss_data_dir=%s, host=%s, bucket_name=%s' %
                 (git_oss_data_dir, host, bucket_name))

  logging.info('git_oss_cache_dir: %s' % git_oss_cache_dir)

  if not os.path.exists(git_oss_cache_dir):
    os.makedirs(git_oss_cache_dir)

  logging.info('git_oss_private_config=%s' % git_oss_private_path)
  if git_oss_private_path is not None and os.path.exists(git_oss_private_path):
    # load oss configs
    with open(git_oss_private_path, 'r') as fin:
      for line_str in fin:
        line_str = line_str.strip()
        line_tok = [x.strip() for x in line_str.split('=') if x != '']
        if line_tok[0] in ['accessid', 'accessKeyID']:
          accessid = line_tok[1]
        elif line_tok[0] in ['accesskey', 'accessKeySecret']:
          accesskey = line_tok[1]
    oss_auth = oss2.Auth(accessid, accesskey)
    oss_bucket = oss2.Bucket(oss_auth, host, bucket_name)
  else:
    logging.info('git_oss_private_path[%s] is not found, read-only mode' %
                 git_oss_private_path)
    # pull only mode
    oss_auth = None
    oss_bucket = None

  if sys.argv[1] == 'push':
    updated = False
    git_bin_arr = recheck_git_bin()
    git_bin_url = load_git_url()
    for leaf_path in git_bin_arr:
      leaf_files = git_bin_arr[leaf_path]
      # empty directory will not be push to oss
      if len(leaf_files) == 0:
        continue
      file_name = path2name(leaf_path)
      new_sig = get_local_sig(leaf_files)
      if new_sig is None:
        continue
      if leaf_path in git_bin_url and git_bin_url[leaf_path][0] == new_sig:
        continue
      # build tar file and push to oss
      file_name_with_sig = file_name + '_' + new_sig
      tar_out_path = '%s/%s.tar.gz' % (git_oss_cache_dir, file_name_with_sig)
      subprocess.check_output(['tar', '-czf', tar_out_path] + leaf_files)
      save_path = '%s/%s' % (git_oss_data_dir, file_name_with_sig)
      oss_bucket.put_object_from_file(save_path, tar_out_path)
      oss_bucket.put_object_acl(save_path, oss2.OBJECT_ACL_PUBLIC_READ)
      git_bin_url[leaf_path] = (new_sig, save_path)
      logging.info('pushed %s' % leaf_path)
      updated = True
    for leaf_path in list(git_bin_url.keys()):
      if leaf_path not in git_bin_arr:
        del git_bin_url[leaf_path]
        logging.info('dropped %s' % leaf_path)
        updated = True
    if updated:
      save_git_url(git_bin_url)
      logging.info('push succeed.')
    else:
      logging.warning('nothing to push')
    subprocess.check_output(['git', 'add', git_bin_url_path])
  elif sys.argv[1] == 'pull':
    # pull images from remote
    any_update = False
    git_bin_arr = load_git_bin()
    git_bin_url = load_git_url()
    for leaf_path in git_bin_arr:
      leaf_files = git_bin_arr[leaf_path]
      if len(leaf_files) == 0:
        if os.path.isfile(leaf_path):
          logging.error('conflicts: %s is a file, but was a dir' % leaf_path)
        elif not os.path.isdir(leaf_path):
          os.makedirs(leaf_path)
        continue
      # newly add files
      if leaf_path not in git_bin_url:
        continue
      file_name = path2name(leaf_path)
      all_file_exist = True
      for tmp in leaf_files:
        if not os.path.exists(tmp):
          all_file_exist = False
      remote_sig = git_bin_url[leaf_path][0]
      if all_file_exist:
        local_sig = get_local_sig(leaf_files)
        if local_sig == remote_sig:
          continue
      else:
        local_sig = ''

      update = False
      if len(sys.argv) > 2 and (sys.argv[2] == '-f' or
                                sys.argv[2] == '--force'):
        update = True
      else:
        if has_conflict(leaf_path, leaf_files):
          update = get_yes_no(
              'update %s using remote file[remote_sig=%s local_sig=%s]?[N/Y]' %
              (leaf_path, remote_sig, local_sig))
        else:
          update = True
      if not update:
        continue
      # pull from remote oss
      remote_path = git_bin_url[leaf_path][1]
      _, file_name_with_sig = os.path.split(remote_path)
      tar_tmp_path = '%s/%s.tar.gz' % (git_oss_cache_dir, file_name_with_sig)
      max_retry = 5
      while max_retry > 0:
        try:
          if not os.path.exists(tar_tmp_path):
            in_cache = False
            if oss_bucket:
              oss_bucket.get_object_to_file(remote_path, tar_tmp_path)
            else:
              url = 'http://%s.%s/%s' % (bucket_name, host, remote_path)
              # subprocess.check_output(['wget', url, '-O', tar_tmp_path])
              if sys.platform.startswith('linux'):
                subprocess.check_output(['wget', url, '-O', tar_tmp_path])
              elif sys.platform.startswith('darwin'):
                subprocess.check_output(['curl', url, '--output', tar_tmp_path])
              elif sys.platform.startswith('win'):
                subprocess.check_output(['curl', url, '--output', tar_tmp_path])
          else:
            in_cache = True
            logging.info('%s is in cache' % file_name_with_sig)
          subprocess.check_output(['tar', '-zxf', tar_tmp_path])
          local_sig = get_local_sig(leaf_files)
          if local_sig == remote_sig:
            break
          if in_cache:
            logging.warning('cache invalid, will download from remote')
            os.remove(tar_tmp_path)
            continue
          logging.warning('download failed, local_sig(%s) != remote_sig(%s)' %
                          (local_sig, remote_sig))
        except subprocess.CalledProcessError as ex:
          logging.error('exception: %s' % str(ex))
        except oss2.exceptions.RequestError as ex:
          logging.error('exception: %s' % str(ex))

        os.remove(tar_tmp_path)
        if accl_endpoint is not None and host != accl_endpoint:
          logging.info('will try accelerate endpoint: %s' % accl_endpoint)
          host = accl_endpoint
          if oss_auth:
            oss_bucket = oss2.Bucket(oss_auth, host, bucket_name)
        max_retry -= 1

      logging.info('%s updated' % leaf_path)
      any_update = True
    if not any_update:
      logging.info('nothing to be updated')
  elif sys.argv[1] == 'add':
    add_path = sys.argv[2]
    if not os.path.exists(add_path):
      raise ValueError('add path %s does not exist' % add_path)
    bin_file_map = {}
    try:
      bin_file_map = load_git_bin()
    except Exception as ex:
      logging.warning('load_git_bin exception: %s' % traceback.format_exc(ex))
      pass
    leaf_dirs = list_leafs(add_path)
    any_new = False
    for leaf_path, leaf_files in leaf_dirs:
      for leaf_file in leaf_files:
        tmp_out = subprocess.check_output(['git', 'ls-files', leaf_file])
        if len(tmp_out.strip()) > 0:
          subprocess.check_output(['git', 'rm', '--cached', leaf_file])
      if leaf_path not in bin_file_map:
        bin_file_map[leaf_path] = leaf_files
        any_new = True
      else:  # check whether the files are the same
        old_leaf_files = bin_file_map[leaf_path]
        if not lst_eq(old_leaf_files, leaf_files):
          bin_file_map[leaf_path] = merge_lst(old_leaf_files, leaf_files)
          any_new = True
    if any_new:
      # write back to .git_bin_path
      save_git_bin(bin_file_map)
      logging.info('added %s' % add_path)
    else:
      logging.info('already add %s' % add_path)
    subprocess.check_output(['git', 'add', '.git_bin_path'])
  elif sys.argv[1] == 'remove':
    del_path = sys.argv[2]
    try:
      bin_file_map = load_git_bin()
    except Exception as ex:
      logging.warning('load_git_bin exception: %s' % traceback.format_exc(ex))
      pass
    leaf_dirs = list_leafs(del_path)
    any_update = False
    for leaf_path, leaf_files in leaf_dirs:
      if leaf_path in bin_file_map:
        for leaf_file in leaf_files:
          if leaf_file in bin_file_map[leaf_path]:
            tmp_id = bin_file_map[leaf_path].index(leaf_file)
            del bin_file_map[leaf_path][tmp_id]
            any_update = True
        if len(bin_file_map[leaf_path]) == 0:
          del bin_file_map[leaf_path]
    if any_update:
      save_git_bin(bin_file_map)
      logging.info('remove %s' % del_path)
  elif sys.argv[1] == 'resolve_conflict':
    git_objs = {}
    with open(git_bin_path, 'r') as fin:
      merge_start = 0
      for line_str in fin:
        if line_str.startswith('<<<<<<<'):
          merge_start = 1
        elif line_str.startswith('======='):
          merge_start = 2
        elif line_str.startswith('>>>>>>>'):
          merge_start = 0
        elif merge_start == 0:
          tmp_obj = json.loads(line_str)
          leaf_name = tmp_obj['leaf_name']
          leaf_file = tmp_obj['leaf_file']
          git_objs[leaf_name] = leaf_file
        elif merge_start == 1:
          tmp_obj = json.loads(line_str)
          leaf_name = tmp_obj['leaf_name']
          leaf_file = tmp_obj['leaf_file']
          git_objs[leaf_name] = leaf_file
        elif merge_start == 2:
          tmp_obj = json.loads(line_str)
          leaf_name = tmp_obj['leaf_name']
          leaf_file = tmp_obj['leaf_file']
          if leaf_name in git_objs:
            union = git_objs[leaf_name]
            for tmp in leaf_file:
              if tmp not in union:
                union.append(tmp)
                logging.info('add %s to %s' % (tmp, leaf_name))
            git_objs[leaf_name] = union
          else:
            git_objs[leaf_name] = leaf_file
        else:
          logging.warning('invalid state: merge_start = %d, line_str = %s' %
                          (merge_start, line_str))
    save_git_bin(git_objs)

    git_bin_url_map = {}
    with open(git_bin_url_path, 'r') as fin:
      merge_start = 0
      for line_str in fin:
        if line_str.startswith('<<<<<<<'):
          merge_start = 1
        elif line_str.startswith('======='):
          merge_start = 2
        elif line_str.startswith('>>>>>>>'):
          merge_start = 0
        elif merge_start in [0, 1, 2]:
          line_json = json.loads(line_str)
          if line_json['leaf_path'] in git_objs:
            git_bin_url_map[line_json['leaf_path']] = (line_json['sig'],
                                                       line_json['remote_path'])
        else:
          logging.warning('invalid state: merge_start = %d, line_str = %s' %
                          (merge_start, line_str))
    save_git_url(git_bin_url_map)
    logging.info('all conflicts fixed.')
  else:
    logging.warning('invalid cmd: %s' % sys.argv[1])
    logging.warning(
        'choices are: %s' %
        ','.join(['push', 'pull', 'add', 'remove', 'resolve_conflict']))
