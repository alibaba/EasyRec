#!/bin/bash
ROOT_URL="http://easyrec.oss-cn-beijing.aliyuncs.com/tools/"

if [ -z "$TMPDIR" ]
then
  TMPDIR="/tmp"
fi

cache_file=$TMPDIR/protoc-3.4.0.tar.gz
if [[ ! -d protoc ]]; then
  if [ ! -e "$cache_file" ]
  then
    if [[ "$(uname)" == "Darwin" ]]; then
      curl ${ROOT_URL}protoc-3.4.0-osx-x86_64.tar.gz -o $cache_file
      flag=$?
    elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
      wget ${ROOT_URL}protoc-3.4.0-linux-x86_64.tar.gz -O $cache_file
      flag=$?
    else
      echo "unknown system $(uname -a)"
      exit 1
    fi
    if [ $flag -ne 0 ]
    then
      echo "Download protoc-3.4.0.tar.gz failed"
      exit 1
    fi
  fi

  mkdir protoc
  tar -xf $cache_file -C protoc
fi

protoc/bin/protoc easy_rec/python/protos/*.proto  --python_out=.

if [ $? -ne 0 ]
then
  exit 1
fi
