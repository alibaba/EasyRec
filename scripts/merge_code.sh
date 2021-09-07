#!/bin/bash

target_dir=$1

if [ ! -d "$target_dir" ]
then
  echo "$target_dir does not exist"
  exit 1
fi

oss_config=$2
if [ ! -e "$oss_config" ]
then
  echo "ossutil config[$oss_config] does not exist"
  exit 1
fi

CP=/usr/bin/cp
if [ ! -e "$CP" ]
then
  echo "$CP does not exist"
  exit 1
fi

OSSUTIL=`which ossutil`
if [ $? -ne 0 ]
then
   echo "ossutil is not find in path"
   exit 1
fi

$CP -rf $target_dir/data ./
$CP -rf $target_dir/docs ./
$CP -rf $target_dir/samples ./
$CP -rf $target_dir/easy_rec ./
$CP -rf $target_dir/pai_jobs ./
$CP -rf $target_dir/requirements ./
$CP -rf $target_dir/requirements.txt ./
$CP -rf $target_dir/setup.cfg ./
$CP -rf $target_dir/setup.py ./

git add easy_rec
git add samples
git add docs
git add pai_jobs
git add requirements
git add requirements.txt
git add setup.cfg setup.py


find easy_rec -name "*.pyc" | xargs rm -rf
find easy_rec  -name "*_pb2.py" | xargs rm -rf
find . -name "*.swp" | xargs rm -rf
find . -name "*.swo" | xargs rm -rf

version=`date +%Y%m%d`
data_name=easy_rec_data_${version}.tar.gz
tar -cvzf $data_name data
$OSSUTIL --config=$oss_config cp $data_name oss://easyrec/data/
sed -i -e "s/data\/easyrec_data\(_[0-9]\+\)\?.tar.gz/data\/easyrec_data_${version}.tar.gz/g" README.md

echo "merge is done, please commit and push your changes."
