#!/bin/bash
ROOT_URL="http://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/tools/"

rm -rf .git/hooks/pre-commit
cp pre-commit .git/hooks/
chmod a+rx .git/hooks/pre-commit

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

#PATH=protoc/bin protoc/bin/protoc  --doc_out=html,index.html:. easy_rec/python/protos/*.proto
#sed -i 's#<p>#<pre>#g;s#</p>#</pre>#g' index.html
