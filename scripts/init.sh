#!/bin/bash

# init pre-commit check hook
rm -rf .git/hooks/pre-commit
cp pre-commit .git/hooks/
chmod a+rx .git/hooks/pre-commit

# compile proto files
source scripts/gen_proto.sh
