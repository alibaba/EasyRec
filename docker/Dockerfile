FROM datascience-registry.cn-beijing.cr.aliyuncs.com/tensorflow/tensorflow:1.15.5

COPY docker/sources_18.04.list /etc/apt/sources.list

# necessary for later commands to take effect
RUN md5sum /etc/apt/sources.list

RUN apt-get update
RUN apt-get install apt-utils inetutils-ping wget curl telnet vim strace libpq-dev curl libsasl2-dev gcc g++ -y

RUN mkdir /EasyRec
COPY requirements /EasyRec/requirements
COPY requirements.txt /EasyRec/
COPY easy_rec /EasyRec/easy_rec/
COPY setup.cfg  /EasyRec/
COPY setup.py /EasyRec/
COPY MANIFEST.in /EasyRec/
COPY README.md /EasyRec/
COPY scripts /EasyRec/scripts


RUN pip3 install pystack-debugger idna kafka-python -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip3 install -r /EasyRec/requirements/runtime.txt   -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip3 install -r /EasyRec/requirements/extra.txt   -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip3 install http://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/graphlearn-1.1.0-cp36-cp36m-linux_x86_64.whl -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip3 install http://easyrec.oss-cn-beijing.aliyuncs.com/releases/pai_automl-0.0.1rc1-py3-none-any.whl -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip3 install http://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/common_io-0.3.0-cp36-cp36m-linux_x86_64.whl -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN cd /EasyRec && python setup.py install
RUN rm -rf /EasyRec
RUN python -c "import easy_rec; easy_rec.help(); import pyhive; import datahub; import kafka"
