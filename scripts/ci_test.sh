#!/usr/bin/env bash

echo "will test pull_request(number=$PULL_REQUEST_NUM)"

# pip install
pip install oss2
pip install -r requirements.txt

# update/generate proto
bash scripts/gen_proto.sh

if [ -n "$PULL_REQUEST_NUM" ]
then
  # check updates
  PYTHONPATH=. python scripts/ci_test_change_files.py --pull_request_num $PULL_REQUEST_NUM --exclude_dir docs
  if [ $? -ne 0 ]
  then
     # there are no code changes related to this test
     echo "::set-output name=ci_test_passed::0"
     exit
  fi
fi

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
