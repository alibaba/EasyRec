# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Called by pai_hpo.py."""

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--sql_path', type=str, help='output sql path', default=None)
  parser.add_argument(
      '--config_path', type=str, help='config path', default=None)
  parser.add_argument(
      '--tables', type=str, help='train_table and test_table', default=None)
  parser.add_argument(
      '--train_tables', type=str, help='train_tables', default=None)
  parser.add_argument(
      '--eval_tables', type=str, help='eval_tables', default=None)
  parser.add_argument(
      '--cluster',
      type=str,
      help='specify tensorflow train jobs cluster parameter',
      default=None)
  parser.add_argument('--bucket', type=str, help='oss bucket', default=None)
  parser.add_argument(
      '--hpo_param_path', type=str, help='hpo param path', default=None)
  parser.add_argument(
      '--hpo_metric_save_path',
      type=str,
      help='hpo metric save path',
      default=None)
  parser.add_argument('--model_dir', type=str, help='model_dir', default=None)
  parser.add_argument('--oss_host', type=str, help='oss endpoint', default=None)
  parser.add_argument('--role_arn', type=str, help='role arn', default=None)
  parser.add_argument(
      '--algo_proj_name',
      type=str,
      help='algorithm project name',
      default='algo_public')
  parser.add_argument(
      '--algo_res_proj', type=str, help='algo resource project', default=None)
  parser.add_argument(
      '--algo_version', type=str, help='algo version', default=None)

  args = parser.parse_args()

  with open(args.sql_path, 'w') as fout:
    fout.write('pai -name easy_rec_ext -project %s\n' % args.algo_proj_name)
    if args.algo_res_proj:
      fout.write('  -Dres_project=%s\n' % args.algo_res_proj)
    else:
      fout.write('  -Dres_project=%s\n' % args.algo_proj_name)
    if args.algo_version:
      fout.write('  -Dversion=%s\n' % args.algo_version)
    fout.write('  -Dconfig=%s\n' % args.config_path)
    fout.write('  -Dcmd=train\n')
    if args.tables:
      fout.write('  -Dtables=%s\n' % args.tables)
    else:
      fout.write('  -Dtrain_tables=%s\n' % args.train_tables)
      fout.write('  -Deval_tables=%s\n' % args.eval_tables)
    fout.write('  -Dcluster=\'%s\'\n' % args.cluster)
    fout.write('  -Darn=%s\n' % args.role_arn)
    fout.write('  -Dbuckets=%s\n' % args.bucket)
    fout.write('  -Dhpo_param_path=%s\n' % args.hpo_param_path)
    fout.write('  -Dhpo_metric_save_path=%s\n' % args.hpo_metric_save_path)
    fout.write('  -Dmodel_dir=%s\n' % args.model_dir)
    fout.write('  -DossHost=%s\n' % args.oss_host)
    fout.write('  -Deval_method=separate;\n')

  print('write to %s' % args.sql_path)
