#!/bin/bash

bash scripts/gen_proto.sh
if [ $? -ne 0 ]
then
  echo "gen proto failed"
  exit 1
fi

export PYTHONPATH=.
version=`python -c "import easy_rec; print(easy_rec.__version__)" | tail -1`
echo "EasyRec Version: $version"

sudo docker build --net=host . -f docker/Dockerfile -t  datascience-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-${version}
