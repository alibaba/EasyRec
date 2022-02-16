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

PYTHONPATH=. python -m easy_rec.python.test.run --list_test_to_file UNIT_TEST_CASE_LIST

for test_name in `cat UNIT_TEST_CASE_LIST`
do
  rm -rf $UnitTestSucceedFlag
  # run test
  PYTHONPATH=. python -m easy_rec.python.test.run --pattern ${test_name}.*
  # for github
  if [ ! -e "$UnitTestSucceedFlag" ]
  then
    echo "::set-output name=ci_test_passed::0"
    exit
  fi
done

# for github
echo "::set-output name=ci_test_passed::1"
rm -rf $UnitTestSucceedFlag
rm -rf UNIT_TEST_CASE_LIST
