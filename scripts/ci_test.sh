#!/usr/bin/env bash

echo "$PULL_REQUEST_CHANGES"
exit 1 

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

PYTHONPATH=. python -m easy_rec.python.test.run  # --pattern export_test.*

# for github
if [ $? -eq 0 ]
then
  echo "::set-output name=ci_test_passed::0"
else
  echo "::set-output name=ci_test_passed::1"
fi
