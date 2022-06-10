#!/bin/bash

bash scripts/gen_proto.sh
if [ $? -ne 0 ]
then
  echo "gen proto failed"
  exit 1
fi

sudo docker build --net=host . -f docker/Dockerfile -t  datascience-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15
