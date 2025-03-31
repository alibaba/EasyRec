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

# sudo docker build --network=host . -f docker/Dockerfile_tf212gpu -t mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12gpu-${version}
sudo docker build --network=host . -f docker/Dockerfile_tf212gpu -t fanyang0801/easyrec:py38-tf2.12gpu-${version}

# next step is to push the image to registry
# sudo docker push fanyang0801/easyrec:py38-tf2.12gpu-${version}