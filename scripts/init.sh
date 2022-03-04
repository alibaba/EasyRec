#!/bin/bash

# init pre-commit check hook
rm -rf .git/hooks/pre-commit
cp pre-commit .git/hooks/
chmod a+rx .git/hooks/pre-commit

# compile proto files
source scripts/gen_proto.sh

if [ $? -ne 0 ]
then
  echo "generate proto failed."
  exit 1
fi

file_name=easyrec_data_20220304.tar.gz

tmp_dir=$TMPDIR

if [ -z "$tmp_dir" ]
then
  tmp_dir="/tmp"
fi

tmp_path="$tmp_dir/$file_name"
if [ ! -e "$tmp_path" ]
then
  wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/$file_name -O $tmp_path
  if [ $? -ne 0 ]
  then
     echo "download data failed"
     exit 1
  fi
fi
tar -zvxf $tmp_path
if [ $? -ne 0 ]
then 
  echo "extract $file_name failed"
  exit 1
fi
