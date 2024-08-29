# Local Tutorial

### 安装EasyRec

我们提供了`本地Anaconda安装`和`Docker镜像启动`两种方式。

有技术问题可加钉钉群：37930014162

#### 本地Anaconda安装

Demo实验中使用的环境为 `python=3.6.8` + `tenserflow=1.12.0`

```bash
conda create -n py36_tf12 python=3.6.8
conda activate py36_tf12
pip install tensorflow==1.12.0
```

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/init.sh
python setup.py install

```

#### Docker镜像启动

Docker的环境为`python=3.6.9` + `tenserflow=1.15.5`

##### 方法一：拉取已上传的镜像（推荐）

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-0.7.4
docker run -td --network host -v /local_path/EasyRec:/docker_path/EasyRec mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-0.7.4
docker exec -it <CONTAINER_ID> bash
```

##### 方法二：自行构建Docker镜像

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/build_docker.sh
sudo docker run -td --network host -v /local_path:/docker_path mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-<easyrec_version>
sudo docker exec -it <CONTAINER_ID> bash
```

注：\<easyrec_version>需匹配当前EasyRec版本。

### 输入数据:

输入一般是csv格式的文件。

#### 示例数据(点击下载)

- train: [dwd_avazu_ctr_deepmodel_train.csv](http://easyrec.oss-cn-beijing.aliyuncs.com/data/dwd_avazu_ctr_deepmodel_train.csv)
- test: [dwd_avazu_ctr_deepmodel_test.csv](http://easyrec.oss-cn-beijing.aliyuncs.com/data/dwd_avazu_ctr_deepmodel_test.csv)
- 示例:

```
1,10,1005,0,85f751fd,c4e18dd6,50e219e0,0e8e4642,b408d42a,09481d60,a99f214a,5deb445a, f4fffcd0,1,0,2098,32,5,238,0,56,0,5
```

- **Note: csv文件不需要有header!!!**

### 启动命令:

#### 配置文件:

[dwd_avazu_ctr_deepmodel_local.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/DeepFM/dwd_avazu_ctr_deepmodel_local.config), 配置文件采用prototxt格式

#### GPU单机单卡:

```bash
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel_local.config
```

- --pipeline_config_path: 训练用的配置文件
- --continue_train: 是否继续训

#### GPU PS训练

- ps跑在CPU上
- master跑在GPU:0上
- worker跑在GPU:1上
- Note: 本地只支持ps, master, worker模式，不支持ps, chief, worker, evaluator模式

```bash
wget https://easyrec.oss-cn-beijing.aliyuncs.com/scripts/train_2gpu.sh
sh train_2gpu.sh dwd_avazu_ctr_deepmodel_local.config
```

#### 评估:

- **Note: 本示例仅仅展示流程，效果无参考价值。**

```bash
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.eval --pipeline_config_path dwd_avazu_ctr_deepmodel_local.config
```

#### 导出:

```bash
CUDA_VISIBLE_DEVICES='' python -m easy_rec.python.export --pipeline_config_path dwd_avazu_ctr_deepmodel_local.config --export_dir dwd_avazu_ctr_export
```

#### CPU训练/评估/导出

不指定CUDA_VISIBLE_DEVICES即可，例如：

```bash
 python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel_local.config
```

### 配置文件:

#### 输入输出

```protobuf
# 训练文件和测试文件
train_input_path: "dwd_avazu_ctr_deepmodel_train.csv"
eval_input_path: "dwd_avazu_ctr_deepmodel_test.csv"
# 模型保存路径
model_dir: "experiments/easy_rec/"
```

#### 数据相关

数据配置具体见：[数据](../feature/data.md)

```protobuf
# 数据相关的描述
data_config {
  # 字段之间的分隔符
  separator: ","
  # 和csv或者odps table里面字段一一对应
  input_fields: {
    input_name: "label"
    input_type: FLOAT
    default_val:""
  }
  ...
  input_fields: {
    input_name: "site_id"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "site_domain"
    input_type: STRING
    default_val:""
  }
}
```

#### 特征相关

特征配置具体见：[特征](../feature/feature.md)

```protobuf
feature_config: {
  features: {
    input_names: "hour"
    # 特征类型
    feature_type: IdFeature
    # embedding向量的dimension
    embedding_dim: 16
    # hash_bucket大小，通过tf.strings.to_hash_bucket将hour字符串映射到0-49的Id
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
    input_names: "site_category"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 100
  }
  features: {
    input_names: "app_id"
    feature_type: IdFeature
    embedding_dim: 32
    hash_bucket_size: 10000
  }
  ...
  features: {
    input_names: "c15"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 500
  }
  features: {
    input_names: "c16"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 500
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

训练配置具体见：[训练](../train.md)

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
  num_steps:1000
}
```

#### 评估相关

评估配置具体见：[评估](../eval.md)

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

#### 参考手册

[EasyRecConfig参考手册](../reference.md)
