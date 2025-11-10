# TF ON YARN

## 一、说明

TensorFlow是google开源的用于人工智能的学习系统，大大的降低了机器学习、深度学开发成本，分析人员、开发人员可以使用TensorFlow提供的多种算法实现自己的想法验证、模型设计等。
TensorFlow on YARN是EMR推出的结合EMR Hadoop大数据处理能力以及TensorFlow深度学习能力，提供用户调度TensorFlow程序在EMR Hadoop之上，进行分布式处理的功能。

## 二、使用说明

说先需要创建[Data Science集群](https://help.aliyun.com/document_detail/170836.html)，目前EMR 3.13.0版本开始支持创建Data Science集群。
Data Science版本的EMR集群支持GPU调度，所以在Core节点，推荐用户选取GPU机器类型。
目前TensorFlow支持的版本是1.8，用户选择想要安装的驱动以及cuDNN版本，EMR管控会将对应的驱动和cuDNN进行自动安装。

## 三、任务提交

目前任务提交还需要通过命令行提交，或者通过EMR-Flow提交任务(正在开发中)。
如果采用命令行提交，提交命令为el_submit, 如下图：

参数说明：

- -t APP_TYPE  提交的任务类型，支持三种类型的任务类型[tensorflow-ps, tensorflow-mpi, standalone]，三种类型要配合后面运行模式使用

> tensorflow-ps使用的是原生TensorFlow ps 类型

> tensorflow-mpi使用的是 uber 开源的基于MPI架构的horovod
> standalone模式是用户将任务调度到YARN集群中启动单机任务，类似于单机运行
> tensorflow-worker多worker模式，适用于MultiWorkerMirroredStrategy

- -a APP_NAME 提交的任务名称，用户可以根据需要起名
- -m MODE 提交的运行时环境，目前支持四种类型运行时环境[local, virtual-env,docker]

> local 使用的是emr-worker上面的python运行环境，所以如果要使用一些第三方python包需要手动在所有机器上进行安装

> docker 使用的是emr-worker上面的docker运行时，tensorflow运行在docker container内

> virtual-env 使用用户上传的python环境，可以在python环境中安装一些不同于worker节点的python库

- -m_arg MODE_ARG 提交的运行时补充参数，如果运行时是docker，则设置为docker镜像名称，如果是virtual-env，则指定python环境文件名称，tar.gz打包
- -x Exit 分布式TensorFlow有些API需要用户手动退出PS，在这种情况下用户可以指定-x选项，当所有worker完成训练并成功后，PS节点自动退出
- -enable_tensorboard 是否在启动训练任务的同时启动TensorBoard
- -log_tensorboard 如果训练同时启动TensorBoard，需要指定TensorBoard日志位置，需要时HDFS上的位置
- -conf CONF Hadoop conf位置，默认可以不设，使用EMR默认配置
- -f FILES 运行TensorFlow所有依赖的文件和文件夹，包含执行脚本，如果设置virtual-env 执行的virtual-env文件。用户可以将所有依赖放置到一个文件夹中，脚本会自动将文件夹按照文件夹层次关系上传到HDFS中
- -pn TensorFlow启动的参数服务器个数
- -pc 每个参数服务器申请的CPU核数
- -pm 每个参数服务器申请的内存大小
- -wn TensorFlow启动的Worker节点个数
- -wc 每个Worker申请的CPU核数
- -wg 每个Worker申请的GPU核数
- -wm 每个Worker申请的内存个数
- -wait_time  获取资源最大等待时间，单位分钟，比如-wait_time 1指的是启动master后最多等待一分钟获取所有资源，否则master失败
- -c COMMAND 执行的命令，如python census.py

进阶选项，用户需要谨慎使用进阶选项，可能造成任务失败

- -wnpg 每个GPU核上同时跑的worker数量(针对tensorflow-ps)
- -ppn 每个GPU核上同时跑的worker数量（针对horovod）
  以上两个选项指的是单卡多进程操作，由于共用一张显卡，需要在程序上进行一定限制，否则会造成显卡OOM。

## DEMO

[Mnist Demo](./mnist_demo.md)
