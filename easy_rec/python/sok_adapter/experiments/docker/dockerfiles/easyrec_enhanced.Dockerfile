#FROM nvcr.io/nvidia/tensorflow:20.08-tf1-py3
FROM nvcr.io/nvidia/tensorflow:22.03-tf1-py3
#FROM tensorflow/tensorflow:2.3.4-gpu
#FROM nvcr.io/nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

# tf 2.3.1 with cuda 11.1
# FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

# TF 2.8 with cuda 11.6
FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/shanghai
RUN apt-get update && apt-get install -y build-essential software-properties-common libssl-dev wget vim tar gdb sudo
RUN apt-get install -y libglib2.0-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 libxcb-shape0

## install cmake
RUN apt remove -y cmake
RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz && tar -zxvf cmake-3.20.2.tar.gz \
    && cd cmake-3.20.2 && ./bootstrap && make -j && make install

## install NCCL
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
#     mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
#     add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
#     apt-get update && \
#     apt install -y libnccl2=2.11.4-1+cuda11.0 libnccl-dev=2.11.4-1+cuda11.0

# Install pip
ADD ./requirements /tmp/requirements
RUN pip3 install -r /tmp/requirements/runtime.txt
RUN pip3 install -r /tmp/requirements/tests.txt
# RUN chmod -R 777 /easyrec_env
RUN chmod 777 /usr/local/lib/python3.8/dist-packages