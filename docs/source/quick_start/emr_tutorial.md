# EMR Tutorial

## 输入数据:

输入一般是csv格式的文件。 如下所示，列之间用,分割

- 示例数据:
  - train: [dwd_avazu_ctr_deepmodel_train.csv](http://easyrec.oss-cn-beijing.aliyuncs.com/data/dwd_avazu_ctr_deepmodel_train.csv)
  - test: [dwd_avazu_ctr_deepmodel_test.csv](http://easyrec.oss-cn-beijing.aliyuncs.com/data/dwd_avazu_ctr_deepmodel_test.csv)
  - 示例:

```
1,10,1005,0,85f751fd,c4e18dd6,50e219e0,0e8e4642,b408d42a,09481d60,a99f214a,5deb445a, f4fffcd0,1,0,2098,32,5,238,0,56,0,5
```

- **Note: csv文件不需要有header!!!**

## 创建DataScience集群:

[DataScience集群](https://help.aliyun.com/document_detail/170836.html)参考

## Copy data to HDFS

```bash
hadoop fs -mkdir -p hdfs://emr-header-1:9000/user/easy_rec/data/
hadoop fs -put dwd_avazu_ctr_deepmodel_train.csv hdfs://emr-header-1:9000/user/easy_rec/data/
hadoop fs -put dwd_avazu_ctr_deepmodel_test.csv hdfs://emr-header-1:9000/user/easy_rec/data/
```

## 训练:

- 配置文件: [dwd_avazu_ctr_deepmodel.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/emr/dwd_avazu_ctr_deepmodel.config) \*\* \*\* 配置文件采用prototxt格式，内容解析见[配置文件](#Qgqxc)
- 使用el_submit提交训练任务，**el_submit**相关参数请参考[**tf_on_yarn**](../tf_on_yarn.md)

### 开源TF模式

`el_submit -yaml train.tf.yaml` 配置文件内容如下

```bash
app:
    app_type: tensorflow-ps
    app_name: easyrec_tf_train
    mode: local
    exit_mode: true
    verbose: true
    files: dwd_avazu_ctr_deepmodel.config
    command: python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config --continue_train
    wait_time: 8
    hook: /usr/local/dstools/bin/hooks.sh

resource:
    ps_num: 1
    ps_cpu: 1
    ps_memory: 10g
    ps_mode_arg:
    worker_num: 2
    worker_cpu: 6
    worker_gpu: 1
    worker_memory: 10g
    worker_mode_arg:

```

### Paitf模式

**使用Paitf需要token授权， 请联系产品架构团队索取token**
`el_submit -yaml train.paitf.yaml `配置文件内容如下

```bash
app:
    app_type: tensorflow-ps
    app_name: easyrec_paitf_train
    mode: docker-pai
    mode_arg: paitf:1.12-gpu
    token: AAAAAAAAAAAAAAABBBBBBBBBBBBBBB==
    exit_mode: true
    verbose: true
    files: dwd_avazu_ctr_deepmodel.config
    command: python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config --continue_train
    hook: /usr/local/dstools/bin/hooks.sh
    wait_time: 8

resource:
    ps_num: 1
    ps_cpu: 1
    ps_memory: 10g
    ps_mode_arg:  paitf:1.12-cpu
    worker_num: 1
    worker_cpu: 6
    worker_gpu: 1
    worker_memory: 10g
    worker_mode_arg: paitf:1.12-gpu

```

- [查看任务日志](../emr_yarn_log.md)

## 评估:

- 使用el_submit提交评估任务，**el_submit**相关参数请参考[**tf_on_yarn**](../tf_on_yarn.md)
- **Note: 本示例仅仅展示流程，效果无参考价值。**

### 开源TF模式

`el_submit -yaml eval.tf.yaml `配置文件内容如下

```bash
app:
    app_type: standalone
    app_name: easyrec_tf_eval
    mode: local
    exit_mode: true
    verbose: true
    files: dwd_avazu_ctr_deepmodel.config
    command: python -m easy_rec.python.eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config
    wait_time: 8
    hook: /usr/local/dstools/bin/hooks.sh

resource:
    worker_num: 1
    worker_cpu: 6
    worker_gpu: 1
    worker_memory: 10g
    worker_mode_arg:

```

### Paitf模式

**使用Paitf需要token授权， 请联系产品架构团队索取token**
`el_submit -yaml eval.paitf.yaml `配置文件内容如下

```bash
app:
    app_type: standalone
    app_name: easyrec_paitf_eval
    mode: docker-pai
    mode_arg: paitf:1.12-gpu
    token: AAAAAAAAAAAAAAABBBBBBBBBBBBBBB==
    exit_mode: true
    verbose: true
    files: dwd_avazu_ctr_deepmodel.config
    command: python -m easy_rec.python.eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config
    wait_time: 8
    hook: /usr/local/dstools/bin/hooks.sh

resource:
    worker_num: 1
    worker_cpu: 6
    worker_gpu: 1
    worker_memory: 10g
    worker_mode_arg: paitf:1.12-gpu

```

## 导出:

- 使用el_submit提交导出任务, **el_submit**相关参数请参考[**tf_on_yarn**](https://help.aliyun.com/document_detail/93031.html)

--pipeline_config_path: EasyRec配置文件
--export_dir: 导出模型目录 
--checkpoint_path: 指定checkpoint，默认不指定，不指定则使用model_dir下面最新的checkpoint

### 开源TF模式

`el_submit -yaml export.tf.yaml `配置文件内容如下

```bash
app:
    app_type: standalone
    app_name: easyrec_tf_export
    mode: local
    exit_mode: true
    verbose: true
    files: dwd_avazu_ctr_deepmodel.config
    command: python -m easy_rec.python.export --pipeline_config_path dwd_avazu_ctr_deepmodel.config --export_dir hdfs://emr-header-1:9000/user/easy_rec/experiment/export
    wait_time: 8
    hook: /usr/local/dstools/bin/hooks.sh

resource:
    worker_num: 1
    worker_cpu: 6
    worker_gpu: 1
    worker_memory: 10g
    worker_mode_arg:

```

### Paitf模式

**使用Paitf需要token授权， 请联系产品架构团队索取token**
`el_submit -yaml export.paitf.yaml `配置文件内容如下

```bash
app:
    app_type: standalone
    app_name: easyrec_paitf_export
    mode: docker-pai
    mode_arg: paitf:1.12-gpu
    token: AAAAAAAAAAAAAAABBBBBBBBBBBBBBB==
    exit_mode: true
    verbose: true
    files: dwd_avazu_ctr_deepmodel.config
    command: python -m easy_rec.python.export --pipeline_config_path dwd_avazu_ctr_deepmodel.config --export_dir hdfs://emr-header-1:9000/user/easy_rec/experiment/export
    wait_time: 8
    hook: /usr/local/dstools/bin/hooks.sh

resource:
    worker_num: 1
    worker_cpu: 6
    worker_gpu: 1
    worker_memory: 10g
    worker_mode_arg: paitf:1.12-gpu

```

### 查看导出结果

```bash
hadoop fs -ls hdfs://emr-header-1:9000/user/easy_rec/experiment/export
```

## 部署到Pai-EAS服务

### 1. 导出模型到savedmodel， 并压缩成tar包

```
hadoop fs -get /user/easy_rec/experiment/export/mazeng/1606721697 savedmodel
tar zcvf savedmodel.tar.gz savedmodel
```

### 2. 配置AK， 部署eas服务需要

```
eascmd64 config -i AAAAAAAAAA -k BBBBBBBBBBB -e pai-eas.cn-beijing.aliyuncs.com
```

### 3. 上传模型压缩包，获取oss url

```
eascmd64 upload savedmodel.tar.gz
```

### 4. 部署到线上服务（慎行）

```
eascmd64 create pmml.json
```

pmml.json配置文件内容如下, easyrec是基于tensorflow/paitf的， 因此processor需选择tensorflow相关的

```bash
{
  "name": "demo0",
  "generate_token": "true",
  "model_path": "oss://eas-model-beijing/166408185111/savedmodel.tar.gz",
  "processor": "tensorflow_cpu_1.14",
  "metadata": {
    "instance": 1,
    "eas.enabled_model_verification": false,
    "cpu": 1
  }
}
```

### 5. 构造服务请求

参考 [https://help.aliyun.com/document_detail/111055.html](https://help.aliyun.com/document_detail/111055.html)

#### 1) 获取模型input output信息

```
curl http://pai-eas-vpc.cn-beijing.aliyuncs.com/api/predict/mnist_saved_model_example | python -mjson.tool
```

#### 2) python版

参考 [https://github.com/pai-eas/eas-python-sdk](https://github.com/pai-eas/eas-python-sdk)

```
#!/usr/bin/env python

from eas_prediction import PredictClient
from eas_prediction import StringRequest
from eas_prediction import TFRequest

if __name__ == '__main__':
    client = PredictClient('http://1828488879222746.cn-beijing.pai-eas.aliyuncs.com', 'mnist_saved_model_example')
    client.set_token('AAAAAAAAAAAAAAABBBBBBBBBBBBBBB==')
    client.init()

    #request = StringRequest('[{}]')
    req = TFRequest('predict_images')
    req.add_feed('images', [1, 784], TFRequest.DT_FLOAT, [1] * 784)
    for x in range(0, 1000000):
        resp = client.predict(req)
        print(resp)

```

#### 3) 其他语言版

参考 [https://help.aliyun.com/document_detail/111055.html](https://help.aliyun.com/document_detail/111055.html)

### 配置文件:

#### 输入输出

```protobuf
# 训练表和测试数据
train_input_path: "hdfs://emr-header-1:9000/user/easy_rec/data/dwd_avazu_ctr_deepmodel_train.csv"
eval_input_path: "hdfs://emr-header-1:9000/user/easy_rec/data/dwd_avazu_ctr_deepmodel_test.csv"
# 模型保存路径
model_dir: "hdfs://emr-header-1:9000/user/easy_rec/experiment/"
```

#### 数据相关

```protobuf
# 数据相关的描述
data_config {
  separator: ","
  input_fields: {
    input_name: "label"
    input_type: FLOAT
    default_val:""
  }
  input_fields: {
    input_name: "hour"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c1"
    input_type: STRING
    default_val:""
  }
  ...
  input_fields: {
    input_name: "c20"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c21"
    input_type: STRING
    default_val:""
  }

  label_fields: "label"

  batch_size: 1024
  prefetch_size: 32
  input_type: CSVInput
}
```

#### 特征相关

```protobuf
feature_config:{
  features: {
    input_names: "hour"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 50
  }
  features: {
    input_names: "c1"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 10
  }
  ...
  features: {
    input_names: "c20"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 500
  }
  features: {
    input_names: "c21"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 500
  }
}

```

#### 训练相关

```protobuf
# 训练相关的参数
train_config {
  # 每200轮打印一行log
  log_step_count_steps: 200
  # 优化器相关的参数
  optimizer_config: {
    adam_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.0001
          decay_steps: 100000
          decay_factor: 0.5
          min_learning_rate: 0.0000001
        }
      }
    }
    use_moving_average: false
  }
  # 使用SyncReplicasOptimizer进行分布式训练(同步模式)
  sync_replicas: true
  # num_steps = total_sample_num * num_epochs / batch_size / num_workers
  num_steps: 2000
}
```

#### 评估相关

```protobuf
eval_config {
  # 仅仅评估1000个样本，这里是为了示例速度考虑，实际使用时需要删除
  num_examples: 1000
  metrics_set: {
    # metric为auc
    auc {}
  }
}
```

#### 模型相关

```protobuf
model_config:{
  model_class: "MultiTower"
  feature_groups: {
    group_name: "item"
    feature_names: "c1"
    feature_names: "banner_pos"
    feature_names: "site_id"
    feature_names: "site_domain"
    feature_names: "site_category"
    feature_names: "app_id"
    feature_names: "app_domain"
    feature_names: "app_category"
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "user"
    feature_names: "device_id"
    feature_names: "device_ip"
    feature_names: "device_model"
    feature_names: "device_type"
    feature_names: "device_conn_type"
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "user_item"
    feature_names: "hour"
    feature_names: "c14"
    feature_names: "c15"
    feature_names: "c16"
    feature_names: "c17"
    feature_names: "c18"
    feature_names: "c19"
    feature_names: "c20"
    feature_names: "c21"
    wide_deep:DEEP
  }

  multi_tower {
    towers {
      input: "item"
      dnn {
        hidden_units: [384, 320, 256, 192, 128]
      }
    }

    towers {
      input: "user"
      dnn {
        hidden_units: [384, 320, 256, 192, 128]
      }
    }

    towers {
      input: "user_item"
      dnn {
        hidden_units: [384, 320, 256, 192, 128]
      }
    }

    final_dnn {
      hidden_units: [256, 192, 128, 64]
    }
    l2_regularization: 0.0
  }
  embedding_regularization: 0.0
}
```

#### Config下载

[dwd_avazu_ctr_deepmodel.config](http://easyrec.oss-cn-beijing.aliyuncs.com/config/emr/dwd_avazu_ctr_deepmodel.config)

#### ExcelConfig下载

ExcelConfig比Config更加简明

- [dwd_avazu_ctr_deepmodel.xls](http://easyrec.oss-cn-beijing.aliyuncs.com/data/dwd_avazu_ctr_deepmodel.xls)
- [ExcelConfig 转 Config](../feature/excel_config.md)

### 参考手册

- [EasyRecConfig参考手册](../reference.md)

- [TF on EMR参考手册](../tf_on_yarn.md)

- [DataScience集群手册](https://help.aliyun.com/document_detail/170836.html)

- [EMR Tensorboard](../emr_tensorboard.md)
