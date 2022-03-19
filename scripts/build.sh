#!/bin/sh

sh -x scripts/gen_proto.sh 
python setup.py sdist bdist_wheel 
ls -lh dist/easy*.whl
