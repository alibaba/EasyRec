#!/bin/sh

cd ../ && sh -x scripts/gen_proto.sh && python3.7 setup.py sdist bdist_wheel && cp package/dist/easy*.whl . && cd -
