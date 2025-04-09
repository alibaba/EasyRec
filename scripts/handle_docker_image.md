# 阿里云容器镜像服务

* Reference: https://cr.console.aliyun.com/repository/cn-hangzhou/fanyang0801/tensorflow/details

1. 登录阿里云Docker Registry

```shell
$ docker login --username=shevapato2017 crpi-sjgh70na3ul3ri0q.cn-hangzhou.personal.cr.aliyuncs.com
```

2. 从Registry中拉取镜像
```shell
$ docker pull crpi-sjgh70na3ul3ri0q.cn-hangzhou.personal.cr.aliyuncs.com/fanyang0801/tensorflow:[镜像版本号]
```

3. 将镜像推送到Registry

```shell
$ docker login --username=shevapato2017 crpi-sjgh70na3ul3ri0q.cn-hangzhou.personal.cr.aliyuncs.com
$ docker tag [ImageId] crpi-sjgh70na3ul3ri0q.cn-hangzhou.personal.cr.aliyuncs.com/fanyang0801/tensorflow:[镜像版本号]
$ docker push crpi-sjgh70na3ul3ri0q.cn-hangzhou.personal.cr.aliyuncs.com/fanyang0801/tensorflow:[镜像版本号]
```

4. 选择合适的镜像仓库地址

从ECS推送镜像时，可以选择使用镜像仓库内网地址。推送速度将得到提升并且将不会损耗您的公网流量。

如果您使用的机器位于VPC网络，请使用 crpi-sjgh70na3ul3ri0q-vpc.cn-hangzhou.personal.cr.aliyuncs.com 作为Registry的域名登录。

5. 示例

使用"docker tag"命令重命名镜像，并将它通过专有网络地址推送至Registry。

```shell
$ docker images
```

```text
REPOSITORY                                                         TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
registry.aliyuncs.com/acs/agent                                    0.7-dfb6816         37bb9c63c8b2        7 days ago          37.89 MB
$ docker tag 37bb9c63c8b2 crpi-sjgh70na3ul3ri0q-vpc.cn-hangzhou.personal.cr.aliyuncs.com/acs/agent:0.7-dfb6816
```

使用 "docker push" 命令将该镜像推送至远程。

```shell
$ docker push crpi-sjgh70na3ul3ri0q-vpc.cn-hangzhou.personal.cr.aliyuncs.com/acs/agent:0.7-dfb6816
```