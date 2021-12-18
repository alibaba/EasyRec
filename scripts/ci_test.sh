#!/usr/bin/env bash

# pip install
pip install oss2
pip install -r requirements.txt

# update/generate proto
bash scripts/gen_proto.sh

export CUDA_VISIBLE_DEVICES=""
export TEST_DEVICES=""

if [[ $# -eq 1 ]]; then
  export TEST_DIR=$1
else
  export TEST_DIR="/tmp/easy_rec_test_${USER}_`date +%s`"
fi

export UnitTestSucceedFlag=EasyRecUnitSucceed
rm -rf $UnitTestSucceedFlag
# run test
PYTHONPATH=. python easy_rec/python/test/run.py 

# for github
if [ -e "$UnitTestSucceedFlag" ]
then
    echo "::set-output name=ci_test_passed::1"
else
    echo "::set-output name=ci_test_passed::0"
fi
