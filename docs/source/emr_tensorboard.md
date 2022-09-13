# EMR tensorboard

1. 在Header上启动tensorboard

```bash
ssh root@39.104.103.119 # login to header
source $HADOOP_HOME/libexec/hadoop-config.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server
CLASSPATH=$($HADOOP_HDFS_HOME/bin/hadoop classpath --glob) tensorboard --logdir=hdfs:///user/experiments/mnist_train_v2 --port 6006
```

2. 通过SSH隧道方式建立代理

- 详细见 [通过SSH隧道方式访问开源组件Web UI](https://help.aliyun.com/document_detail/169151.html?spm=a2c4g.11186623.6.598.658d727beowT5O)

```bash
# 在mac上执行
ssh -N -D 8157 root@39.104.103.119
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome  --proxy-server="socks5://localhost:8157" --host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost" --user-data-dir=/tmp/
```

3. 在浏览器中输入: [http://emr-header-1:6006/](http://emr-header-1:6006/)
