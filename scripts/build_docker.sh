#!/bin/bash

bash scripts/gen_proto.sh
if [ $? -ne 0 ]
then
  echo "gen proto failed"
  exit 1
fi

version=`grep "__version__" easy_rec/version.py | awk '{ if($1 == "__version__") print $NF}'`
# strip "'"
version=${version//\'/}
echo "EasyRec Version: $version"

if [ -z "$version" ]
then
  echo "Failed to get EasyRec version"
  exit 1
fi

sudo docker build --network=host . -f docker/Dockerfile -t  mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-${version}
