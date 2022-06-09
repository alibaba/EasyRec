#!/bin/bash
sudo docker build --net=host . -f docker/Dockerfile -t  datascience-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15
