#!/bin/bash
ROOT_URL="http://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/tools/"

if [[ ! -d protoc ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    curl ${ROOT_URL}protoc-3.4.0-osx-x86_64.tar.gz -o protoc-3.4.0.tar.gz
  elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    wget ${ROOT_URL}protoc-3.4.0-linux-x86_64.tar.gz -O protoc-3.4.0.tar.gz
  fi
  mkdir protoc
  tar -xf protoc-3.4.0.tar.gz -C protoc
fi
protoc/bin/protoc easy_rec/python/protos/*.proto  --python_out=.
if [ $? -ne 0 ]
then
  exit 1
fi

cp easy_rec/python/protos/dataset_pb2.py  easy_rec/python/protos/tf_predict_pb2.py processor/
