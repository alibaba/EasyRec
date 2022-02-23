#!/usr/bin/env bash

# pip install
pip install oss2
pip install -r requirements.txt

# update/generate proto
bash scripts/gen_proto.sh

export CUDA_VISIBLE_DEVICES=""
export TEST_DEVICES=""

export TEST_DIR="/tmp/easy_rec_test_${USER}_`date +%s`"

if [ -e UNIT_TEST_CASE_LIST ]; then
  test_patterns=$(awk '{print $0".*"}' UNIT_TEST_CASE_LIST)
else
  test_patterns=()
fi

while getopts "d:t:" opt; do
  case $opt in
    d)
      export TEST_DIR="$OPTARG"
      ;;
    t)
      test_patterns=($OPTARG)
      ;;
  esac
done

export UnitTestSucceedFlag=EasyRecUnitSucceed

rm -rf $UnitTestSucceedFlag
for test_pattern in ${test_patterns[@]}
do
  rm -rf $UnitTestSucceedFlag
  echo "running unittest: ${test_pattern}"
  # run test
  PYTHONPATH=. python -m easy_rec.python.test.run --pattern ${test_pattern}
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
