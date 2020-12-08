#!/usr/bin/env bash

# pip install
pip install oss2
pip install -r requirements.txt

# setup for git-lfs
if [[ ! -e git-lfs/git_lfs.py ]]; then
  git submodule init
  git submodule update
fi

# download test data
python git-lfs/git_lfs.py pull

# update/generate proto
bash scripts/gen_proto.sh

export CUDA_VISIBLE_DEVICES=""

if [[ $# -eq 1 ]]; then
  export TEST_DIR=$1
else
  export TEST_DIR="/tmp/easy_rec_test_${USER}_`date +%s`"
fi

# run test
PYTHONPATH=. python easy_rec/python/test/run.py
