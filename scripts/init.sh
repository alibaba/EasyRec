#!/bin/bash

# init pre-commit check hook
rm -rf .git/hooks/pre-commit
cp pre-commit .git/hooks/
chmod a+rx .git/hooks/pre-commit

python git-lfs/git_lfs.py pull

# compile proto files
source scripts/gen_proto.sh
