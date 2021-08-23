# Mnist Demo on EMR

本示例中的程序都可以在**tf1.15**或者**tf2.0**上运行

## 单机多卡模式: MirroredStragy

使用keras model，是tf2.x推荐运行的方式

```bash
wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/mnist_demo/mnist.npz
hadoop fs -mkdir -p hdfs:///user/data/
hadoop fs -put mnist.npz hdfs:///user/data/

wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/mnist_demo/mnist_mirrored.py -O mnist_mirrored.py
把strategy = tf.distribute.MirroredStrategy()
替换成strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

el_submit  -t standalone -a mnist_train -f mnist_mirrored.py  -m local -wn 1 -wg 2  -wc 6  -wm 20000 -c python mnist_mirrored.py
```

- -wn: worker number，必须是1
- -wg: 2, 2GPUS
- -wc: CPU number
- -wm: cpu memory size in bytes, 20000 is 20G

## 多机多卡模式: MultiWorkerMirroredStrategy

```bash
wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/mnist_demo/mnist.npz
hadoop fs -mkdir -p hdfs:///user/data/
hadoop fs -put mnist.npz hdfs:///user/data/
wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/mnist_demo/mnist_mirrored.py -O mnist_mirrored.py

el_submit  -t tensorflow-worker -a mnist_train -f mnist_mirrored.py  -m local -wn 2 -wg 1  -wc 6  -wm 20000 -c python mnist_mirrored.py
```

- -wn: worker number，2
- -wg: 1, 1GPU, 可以 > 1
- -wc: CPU number
- -wm: cpu memory size in bytes, 20000 is 20G
