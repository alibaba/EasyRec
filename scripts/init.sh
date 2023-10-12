#!/bin/bash

# init pre-commit check hook
rm -rf .git/hooks/pre-commit
cp scripts/git/pre-commit .git/hooks/
chmod a+rx .git/hooks/pre-commit

rm -rf .git/hooks/post-checkout
cp scripts/git/post-checkout .git/hooks/
chmod a+rx .git/hooks/post-checkout

python git-lfs/git_lfs.py pull

# compile proto files
source scripts/gen_proto.sh
